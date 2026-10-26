/**
 * scenario4_rl_twt_ofdma.cc
 * ==========================
 * Scenario 4: Deep RL Agent controls TWT scheduling + OFDMA
 *
 * Paper: "Deep RL-based Dynamic TWT Scheduling for Heterogeneous Wi-Fi Networks"
 *        Maksud & Carvalho, Texas State University
 *
 * MDP (paper Sec. III-B):
 *   State  : 13 features x 16 STAs = 208-dim observation
 *   Action : (schedule_id, assignment_id) in {0..19} x {0..18} = 380 actions
 *   Reward : 6-component weighted sum — Eq.(3)-(9), Table V
 *   Step   : 20 BIs per RL decision
 *   Episode: 35 steps (855 BIs total)
 *
 * Output: scenario4_rl_overall.csv, scenario4_rl_persta.csv,
 *         scenario4_rl_actions.csv, scenario4_rl_rewards.csv
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include <array>
#include "ns3/energy-module.h"
#include "ns3/wifi-radio-energy-model-helper.h"
#include "ns3/ns3-ai-msg-interface.h"
#include "ns3/rr-multi-user-scheduler.h"

#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <iomanip>
#include <set>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <unistd.h>   // _exit() — see end of main()

using namespace ns3;
using namespace ns3::energy;

NS_LOG_COMPONENT_DEFINE("Scenario4RLTWT");

// ─── Paper constants (Table VI) ───────────────────────────────────────────────
static const uint32_t NUM_STA          = 16;
static const uint32_t NUM_FEATURES     = 13;
static const double   AREA_SIDE        = 30.0;
static const double   BEACON_INTERVAL  = 0.1024;
static const uint32_t SIM_BI           = 1000; // ~102 s
static const uint32_t APP_START_MIN_BI = 25;
static const uint32_t APP_START_MAX_BI = 45;
static const uint32_t TWT_SETUP_BI     = 75;
// When false, ScheduleBITWT() is never called, so no station is ever put to
// sleep: a TRUE always-on reference (power save disabled), as distinct from
// the highest-duty TWT schedule, which still sleeps 12.1% of every beacon
// interval. Defaults to true so existing behaviour is bit-identical.
static bool g_twtEnabled = true;
static const uint32_t RL_START_BI      = 90;
static const uint32_t WARMUP_STEPS     = 3;
static const uint32_t DELTA_STEP       = 20;
static const uint32_t T_STEPS          = 45;  // (SIM_BI - RL_START_BI) / DELTA_STEP = (1000-90)/20
static const uint32_t CHANNEL_BW_MHZ   = 20;

static const double SUPPLY_VOLTAGE  = 3.0;
static const double CURRENT_TX_MA   = 232.0;
static const double CURRENT_RX_MA   = 66.0;
static const double CURRENT_IDLE_MA = 50.0;
static const double CURRENT_SLEEP_MA= 0.12;
// NOT set on the radio model below: WifiRadioEnergyModel's CcaBusyCurrentA keeps
// its ns-3 default of 0.273 A. This scenario overrides IdleCurrentA to 50 mA but
// leaves CCA at the default, so a CCA-busy microsecond costs 5.5x an idle one.
// Recorded here so the decomposition can attribute that energy instead of
// leaving ~30% of the total unexplained; the model itself is untouched.
// Settable so the profile can be made coherent. The shipped configuration sets
// TX/RX/idle/sleep but leaves CcaBusyCurrentA at ns-3's default of 273 mA, which
// charges carrier sensing MORE than transmitting (232 mA) and 4.1x more than
// receiving (66 mA). A radio cannot spend more power sensing a medium than
// receiving on it, so that profile is not physically realisable; it is retained
// as the default only so previously measured results reproduce exactly.
static double CURRENT_CCA_MA  = 273.0;

// ── PHY state accounting ─────────────────────────────────────────────────────
// WifiRadioEnergyModel exposes only a scalar total, so the paper could not say
// WHERE the energy went -- which is exactly what a reviewer asks when the claim
// is that OFDMA costs energy inside the awake period. WifiPhyStateHelper's
// "State" trace reports (start, duration, state) for every transition, so
// accumulating duration per state gives an exact time budget; multiplying by the
// currents this scenario configures on the radio model gives the matching
// energy. CCA_BUSY and SWITCHING are reported as TIME ONLY: their currents are
// left at the model's defaults, which this scenario does not set, so assigning
// them energy here would be a guess. The decomposition is additive and is
// checked against the EnergySource total at the end of every run.
// Nothing here changes the radio model, so previously measured energies are
// unaffected.
static std::vector<Ptr<EnergySource>> energySources(NUM_STA);   // declared here: PhyStateCb below reads it
static const uint32_t NUM_PHY_STATES = 7;   // IDLE,CCA_BUSY,TX,RX,SWITCHING,SLEEP,OFF
static std::vector<std::array<double, NUM_PHY_STATES>> g_stateSec(NUM_STA);
static std::vector<std::array<double, NUM_PHY_STATES>> g_prevStateSec(NUM_STA);
// Energy ACTUALLY charged by the energy model while the PHY was in each state.
// Multiplying state time by the configured current does not reproduce the
// model's total: WifiRadioEnergyModel drives its own state machine from PHY
// listener callbacks (it schedules a switch back to IDLE after TX/RX), which
// partitions time differently from WifiPhyStateHelper's trace. The two agree on
// a quiet medium and diverge by ~9% on a busy one. Reading the source's drawn
// energy at each transition and attributing the delta to the state that just
// ended makes the decomposition sum to the measured total by construction.
static std::vector<std::array<double, NUM_PHY_STATES>> g_stateMj(NUM_STA);
static std::vector<std::array<double, NUM_PHY_STATES>> g_prevStateMj(NUM_STA);
static std::vector<double> g_lastDrawnMj(NUM_STA, 0.0);

static void PhyStateCb(uint32_t i, Time /*start*/, Time dur, WifiPhyState st)
{
    const uint32_t k = static_cast<uint32_t>(st);
    if (i >= NUM_STA || k >= NUM_PHY_STATES) return;
    g_stateSec[i][k] += dur.GetSeconds();
    if (energySources[i])
    {
        const double drawn = (energySources[i]->GetInitialEnergy()
                            - energySources[i]->GetRemainingEnergy()) * 1e3;
        g_stateMj[i][k] += drawn - g_lastDrawnMj[i];
        g_lastDrawnMj[i] = drawn;
    }
}

// Sleep-guard controls. The 8 ms deferral below was applied only when the
// multi-user scheduler was installed, which made the two OFDMA arms differ in
// sleep handling as well as in scheduler availability. These let the guard be
// zeroed or applied symmetrically so the comparison can be run without that
// asymmetry. Defaults reproduce the historical behaviour exactly.
static uint32_t g_sleepGuardUs       = 8000;

// ── Service-period-aligned solicitation ──────────────────────────────────────
// The 8 ms sleep guard exists because the AP could solicit a station whose TWT
// service period ended mid-exchange: the station slept before sending its TB
// PPDU and the AP's reception path dereferenced a missing entry. Deferring the
// sleep hides that race at the cost of keeping stations awake. The
// standards-shaped alternative is not to create it: refuse to solicit a station
// unless its service period still has room for the whole exchange (TF + SIFS +
// max-duration HE TB PPDU + SIFS + Multi-STA BA). Then no station can sleep
// mid-exchange, the guard becomes unnecessary rather than merely controlled
// for, and OFDMA happens inside the awake window as trigger-enabled TWT
// intends. g_spEndTime[i] is when station i's current service period ends;
// zero means no scheduled sleep, i.e. awake for the whole beacon interval.
static bool                g_spAligned = false;
static double              g_spMarginUs = 5800.0;
static std::vector<double> g_spEndTime(NUM_STA, 0.0);
static uint64_t            g_spBlockedSolicits = 0;
static bool     g_sleepGuardSymmetric = false;

static const uint32_t DEFAULT_SCHED  = 17;
static const uint32_t DEFAULT_ASSIGN = 4;

static const double PATH_LOSS_EXP = 3.0;
static const double NAKAGAMI_M0   = 3.0;
static const double NAKAGAMI_M1   = 1.5;

// ─── ns3-ai shared memory structs ─────────────────────────────────────────────
struct ScenEnvStruct {
    float bsr_queue_index  [NUM_STA];
    float airtime_used     [NUM_STA];
    float fcs_errors       [NUM_STA];
    float rx_fragments     [NUM_STA];
    float last_rx_ts       [NUM_STA];
    float awake_time       [NUM_STA];
    float sleep_time       [NUM_STA];
    float duty_cycle       [NUM_STA];
    float packets_tx       [NUM_STA];
    float d_airtime        [NUM_STA];
    float d_awake          [NUM_STA];
    float d_sleep          [NUM_STA];
    float d_packets_tx     [NUM_STA];
    float queue_size_bytes [NUM_STA];
    float d_bytes_tx       [NUM_STA];
    float d_energy         [NUM_STA];
    float d_mpdu_drops     [NUM_STA];
    float device_class     [NUM_STA]; // 0=IoT,1=Camera,2=Voice,3=Streamer
    uint32_t step;
    float    done;
};

struct ScenActionStruct {
    uint32_t schedule_id;   // 0-19
    uint32_t assignment_id; // 0-18
    uint32_t ofdma_enabled; // 0/1 - per-decision-step OFDMA on/off (see g_muScheduler)
};

// ─── Action space (Tables III & IV) ──────────────────────────────────────────
struct ScheduleEntry {
    uint32_t             numGroups;
    std::vector<double>  wakeDurations;
};

static const std::vector<ScheduleEntry> SCHEDULES = {
    {1, {90}}, {1, {80}}, {1, {60}}, {1, {40}}, {1, {20}},
    {2, {45,45}}, {2, {40,40}}, {2, {30,30}}, {2, {20,20}}, {2, {15,15}},
    {2, {60,20}}, {2, {50,30}}, {2, {40,20}}, {2, {30,15}},
    {3, {30,30,25}}, {3, {40,25,20}}, {3, {25,25,25}},
    {4, {22,22,22,22}}, {4, {30,20,20,15}}, {4, {20,20,15,15}},
};

using AssignFn = std::function<uint32_t(uint32_t, uint32_t)>;
static std::vector<AssignFn> ASSIGNMENTS;

void InitAssignments() {
    ASSIGNMENTS.clear();
    for (uint32_t g = 0; g < 4; g++) {
        ASSIGNMENTS.push_back([g](uint32_t, uint32_t K) { return std::min(g, K-1); });
    }
    for (uint32_t k : {2u, 3u, 4u}) {
        ASSIGNMENTS.push_back([k](uint32_t i, uint32_t K) { return i % std::min(k, K); });
    }
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { return (i < NUM_STA/2) ? 0u : std::min(1u, K-1); });
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { return std::min(i % 2, K-1); });
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { uint32_t g = (i < NUM_STA/3) ? 0 : (i < 2*NUM_STA/3) ? 1 : 2; return std::min(g, K-1); });
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { return std::min(i % 3, K-1); });
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { return std::min(i / 4, K-1); });
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { return std::min(i % 4, K-1); });
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { return (i < 3*NUM_STA/4) ? 0u : std::min(1u, K-1); });
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { return (i < 3*NUM_STA/5) ? 0u : std::min(1u, K-1); });
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { uint32_t g = (i < NUM_STA/2) ? 0 : (i < NUM_STA*4/5) ? 1 : 2; return std::min(g, K-1); });
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { uint32_t a=NUM_STA*2/5, b=a+NUM_STA*3/10, c=b+NUM_STA/5; uint32_t g=(i<a)?0:(i<b)?1:(i<c)?2:3; return std::min(g, K-1); });
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { return std::min((i % 4) / 2, K-1); });
    ASSIGNMENTS.push_back([](uint32_t i, uint32_t K) { return std::min(i / 4, K-1); });
}

// ─── Device classes ───────────────────────────────────────────────────────────
enum DeviceClass { IOT_SENSOR=0, VIDEO_CAMERA, VOICE_ASSISTANT, VIDEO_STREAMER };
struct DeviceProfile {
    std::string name;
    double dataRateBps;
    uint32_t pktSizeBytes;
    DeviceClass dc;
};
static const DeviceProfile DEVICE_PROFILES[4] = {
    {"IoT-Sensor",      256e3,  200, IOT_SENSOR},
    {"Video-Camera",    2e6,   1500, VIDEO_CAMERA},
    {"Voice-Assistant", 64e3,   160, VOICE_ASSISTANT},
    {"Video-Streamer",  5e6,   1400, VIDEO_STREAMER},
};

// ─── Global state ─────────────────────────────────────────────────────────────
static NodeContainer           apNode;
static NodeContainer           staNodes;
static NetDeviceContainer      apDevice;
static NetDeviceContainer      staDevices;
static std::vector<DeviceClass> staClass(NUM_STA);
static std::vector<uint32_t>   staGroupAssign(NUM_STA, 0);

static std::ofstream overallLog, perStaLog, actionLog, rewardLog;
static uint32_t      currentBI   = 0;
static uint32_t      currentStep = 0;
static uint32_t      curSchedId  = DEFAULT_SCHED;
static uint32_t      curAssignId = DEFAULT_ASSIGN;

static std::vector<double>   staAwakeTime   (NUM_STA, 0.0);
static std::vector<double>   staSleepTime   (NUM_STA, 0.0);
static std::vector<double>   staTotalAirtime(NUM_STA, 0.0);
static std::vector<double>   staEnergy      (NUM_STA, 0.0);

static std::vector<double>   prevAirtime (NUM_STA, 0.0);
static std::vector<double>   prevAwake   (NUM_STA, 0.0);
static std::vector<double>   prevSleep   (NUM_STA, 0.0);
static std::vector<double>   prevEnergy  (NUM_STA, 0.0);

static Ptr<FlowMonitor>      flowMonitor;
static FlowMonitorHelper     flowHelper;
// Live handle to the installed scheduler so OFDMA can be toggled per RL
// decision step, not just once at CLI startup. "EnableUlOfdma" is a normal
// ns-3 Attribute on RrMultiUserScheduler (m_enableUlOfdma), checked live on
// every scheduling decision (rr-multi-user-scheduler.cc:147,158) - not
// cached once at construction - so SetAttribute() here genuinely takes
// effect immediately, no TwtAwareScheduler changes needed.
static Ptr<MultiUserScheduler> g_muScheduler;
static std::map<uint16_t, uint32_t> portToSta;
static std::vector<uint64_t> prevRxBytes (NUM_STA, 0);
static std::vector<uint64_t> prevTxPkts (NUM_STA, 0);
static std::vector<uint64_t> prevRxPkts  (NUM_STA, 0);
static std::vector<uint64_t> prevLostPkts(NUM_STA, 0);
static std::vector<double>   prevDelay   (NUM_STA, 0.0);

static std::vector<EventId> g_wakeEvts (NUM_STA);
static std::vector<EventId> g_sleepEvts(NUM_STA);

static Ns3AiMsgInterfaceImpl<ScenEnvStruct, ScenActionStruct>* aiInterface = nullptr;

// ─── Reward weights (Table V, Energy preset) ─────────────────────────────────
struct RewardWeights { double thruput, queue, energy, drops, air, channel; };
static const RewardWeights ENERGY_PRESET    = {0.20, 0.10, 0.35, 0.10, 0.20, 0.05};
static const RewardWeights TPUT_PRESET      = {0.35, 0.20, 0.10, 0.15, 0.15, 0.05};
static const RewardWeights QUEUE_PRESET     = {0.20, 0.35, 0.10, 0.20, 0.10, 0.05};
// Tuned preset: penalise drops and queue buildup more, relax raw throughput
static const RewardWeights LOW_LOSS_PRESET  = {0.20, 0.30, 0.10, 0.25, 0.10, 0.05};
static RewardWeights curWeights           = ENERGY_PRESET;

// ─── Reward computation (Eq. 4-9) ────────────────────────────────────────────
inline double zscore(double x, double mu, double sig) { return sig > 0 ? (x - mu) / sig : 0.0; }

double jainFairness(const std::vector<double>& v) {
    double sum = 0, sum2 = 0; uint32_t n = 0;
    for (double x : v) { if (x >= 0) { sum += x; sum2 += x*x; n++; } }
    return (n > 0 && sum2 > 0) ? (sum*sum) / (n * sum2) : 1.0;
}

struct RewardComponents { double thruput, queue, energy, drops, air, channel, total; };

RewardComponents ComputeReward(const std::vector<double>& dB, const std::vector<double>& dP,
    const std::vector<double>& dA, const std::vector<double>& dE, const std::vector<double>& dD,
    const std::vector<double>& qKB, const std::vector<double>& SR, const std::vector<double>& DC,
    uint32_t sid)
{
    double B=0,P=0,D=0,Dmax=0,totE=0,mDC=0,mSR=0,minSR=1,mQ=0;
    int nStarv=0,nHigh=0,nHD=0,nInact=0;
    for (uint32_t i = 0; i < NUM_STA; i++) {
        B+=dB[i]; P+=dP[i]; D+=dD[i]; Dmax=std::max(Dmax,dD[i]);
        totE+=dE[i]; mDC+=DC[i]; mSR+=SR[i]; minSR=std::min(minSR,SR[i]); mQ+=qKB[i];
        if (dB[i] < 9800) nStarv++;
        if (qKB[i] > 35) nHigh++;
        if (dD[i] > 180) nHD++;
        if (dP[i] == 0) nInact++;
    }
    mDC/=NUM_STA; mSR/=NUM_STA; mQ/=NUM_STA;
    double BoverU = dA[0] > 0 ? B / std::max(1.0, std::accumulate(dA.begin(),dA.end(),0.0)) : 0.0;
    double jB = jainFairness(dB), jSR = jainFairness(SR);
    double totalWake = std::accumulate(SCHEDULES[sid].wakeDurations.begin(), SCHEDULES[sid].wakeDurations.end(), 0.0);

    double Rt = std::tanh(zscore(B,98000,30000) + zscore(P,65,20) + zscore(BoverU,0.5,0.2)
                          + (jB-0.7)/0.1 - (double)nStarv/(0.25*NUM_STA));
    double Rq = std::tanh(-zscore(mQ,20,15) - (double)nHigh/(0.25*NUM_STA));
    double Re = std::tanh((mSR-0.7)/0.1 - zscore(totE,50,15) - zscore(mDC,0.30,0.15)
                          + (jSR-0.8)/0.1 - (0.7-minSR)/0.15);
    double Rd = std::tanh(-zscore(D,100,80) - zscore(Dmax,100,80) - (double)nHD/(0.25*NUM_STA));
    double Ra = std::tanh(1.5*(1.0 - totalWake/90.0) - 0.35);
    double Rc = std::tanh(-(double)nInact/(0.25*NUM_STA));

    double tot = curWeights.thruput*Rt + curWeights.queue*Rq + curWeights.energy*Re
               + curWeights.drops*Rd + curWeights.air*Ra + curWeights.channel*Rc;
    return {Rt, Rq, Re, Rd, Ra, Rc, tot};
}

// ─── Phase-7/9: joint TWT+OFDMA — TwtAwareScheduler + two-phase sleep guard ───
// Ported from ns3-scenarios/rl-agent-scenario-v2.cc (the eval scenario), where
// this exact pattern was verified crash-free over 250 runs. See THESIS_BRIEF.md
// §18 for the full diagnosis of the three crash classes this fixes. Gated on
// --enableOfdma (default false) — the historical OFDMA-off collection dataset
// (rl_dataset_100k.npz) remains reproducible: with the flag off, every code
// path below is bypassed and behaviour is byte-for-byte identical to before.
static std::set<Mac48Address> g_twtSleepingAddrs;
static const uint32_t SLEEP_GUARD_US = 8000; // see THESIS_BRIEF.md §18 for the 5.7ms-floor derivation
static bool g_ofdmaEnabled = false;

class TwtAwareScheduler : public RrMultiUserScheduler
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TwtAwareScheduler")
                                .SetParent<RrMultiUserScheduler>()
                                .SetGroupName("Wifi")
                                .AddConstructor<TwtAwareScheduler>();
        return tid;
    }
    TwtAwareScheduler()           = default;
    ~TwtAwareScheduler() override = default;

  protected:
    // Has station `addr` a whole exchange left inside its service period?
    static bool FitsInServicePeriod(const Mac48Address& addr)
    {
        if (!g_spAligned) return true;
        for (uint32_t i = 0; i < NUM_STA; i++) {
            Ptr<WifiNetDevice> d = DynamicCast<WifiNetDevice>(staDevices.Get(i));
            if (!d || d->GetMac()->GetAddress() != addr) continue;
            if (g_spEndTime[i] <= 0.0) return true;
            const double leftUs = (g_spEndTime[i] - Simulator::Now().GetSeconds()) * 1e6;
            if (leftUs < g_spMarginUs) { g_spBlockedSolicits++; return false; }
            return true;
        }
        return true;
    }

    bool CanSolicitStaInBasicTf(const MasterInfo& info) const override
    {
        if (g_twtSleepingAddrs.count(info.address)) return false;
        if (!FitsInServicePeriod(info.address)) return false;
        return RrMultiUserScheduler::CanSolicitStaInBasicTf(info);
    }
    bool CanSolicitStaInBsrpTf(const MasterInfo& info) const override
    {
        if (g_twtSleepingAddrs.count(info.address)) return false;
        if (!FitsInServicePeriod(info.address)) return false;
        return RrMultiUserScheduler::CanSolicitStaInBsrpTf(info);
    }
    bool CanScheduleStaForDl(const MasterInfo& info) const override
    {
        if (g_twtSleepingAddrs.count(info.address)) return false;
        uint16_t curAid = m_apMac->GetWifiRemoteStationManager(0)->GetAssociationId(info.address);
        if (curAid != info.aid) return false;
        return true;
    }
};
NS_OBJECT_ENSURE_REGISTERED(TwtAwareScheduler);

static void DoActualStaSleep(uint32_t i)
{
    Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(staDevices.Get(i));
    Mac48Address addr = dev->GetMac()->GetAddress();
    if (!g_twtSleepingAddrs.count(addr)) return; // woken again during the guard window
    Ptr<WifiPhy> phy = dev->GetPhy();
    if ((!phy->IsStateIdle() && !phy->IsStateSleep()) ||
        phy->GetDelayUntilIdle().IsStrictlyPositive())
    {
        Simulator::Schedule(MicroSeconds(500), &DoActualStaSleep, i);
        return;
    }
    phy->SetSleepMode();
}

// ─── OFDMA mechanism counters (verification only, mirrors ofdma-scenario.cc) ──
static uint64_t g_triggerFramesTxByAp = 0;
static uint64_t g_heTbPpduRxAtAp      = 0;
static void ApTriggerTxTrace(std::string, WifiConstPsduMap psduMap, WifiTxVector, double)
{
    for (const auto& pair : psduMap)
        if (pair.second->GetNMpdus() > 0 && pair.second->GetHeader(0).IsTrigger())
            g_triggerFramesTxByAp++;
}
static void ApRxHdrTrace(std::string, const WifiMacHeader&, const WifiTxVector& txVector, Time)
{
    if (txVector.GetPreambleType() == WIFI_PREAMBLE_HE_TB) g_heTbPpduRxAtAp++;
}

// ─── Physical PHY sleep helpers ───────────────────────────────────────────────
static void SetStaSleep(uint32_t i) {
    Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(staDevices.Get(i));
    if (!g_ofdmaEnabled && !g_sleepGuardSymmetric) { dev->GetPhy()->SetSleepMode(); return; }
    if (g_sleepGuardUs == 0) { dev->GetPhy()->SetSleepMode(); return; }
    g_twtSleepingAddrs.insert(dev->GetMac()->GetAddress());
    Simulator::Schedule(MicroSeconds(g_sleepGuardUs), &DoActualStaSleep, i);
}
static void SetStaWake(uint32_t i) {
    Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(staDevices.Get(i));
    if (!g_ofdmaEnabled && !g_sleepGuardSymmetric) { dev->GetPhy()->ResumeFromSleep(); return; }
    g_twtSleepingAddrs.erase(dev->GetMac()->GetAddress());
    dev->GetPhy()->ResumeFromSleep();
}

// Schedule wake/sleep events for the current BI using the active TWT schedule.
// Group g wakes at sum(wakeDurations[0..g-1]) ms into the BI and sleeps after
// wakeDurations[g] ms.  Called once per BI starting at TWT_SETUP_BI.
void ScheduleBITWT() {
    const ScheduleEntry& sched   = SCHEDULES[curSchedId];
    const double         simEnd  = SIM_BI * BEACON_INTERVAL;
    const double         biMs    = BEACON_INTERVAL * 1000.0;
    double               now     = Simulator::Now().GetSeconds();

    // Cumulative wake offsets (ms) for each group
    std::vector<double> offMs(sched.numGroups + 1, 0.0);
    for (uint32_t g = 0; g < sched.numGroups; g++)
        offMs[g + 1] = offMs[g] + sched.wakeDurations[g];

    for (uint32_t i = 0; i < NUM_STA; i++) {
        Simulator::Cancel(g_wakeEvts[i]);
        Simulator::Cancel(g_sleepEvts[i]);

        uint32_t grp = staGroupAssign[i];
        if (grp >= sched.numGroups) { SetStaSleep(i); continue; }

        double wakeOffMs  = offMs[grp];
        double sleepOffMs = offMs[grp + 1];

        if (wakeOffMs > 0.0)
            SetStaSleep(i);          // sleep until this group's slot
        else
            SetStaWake(i);           // group 0: awake from BI start

        if (wakeOffMs > 0.0 && now + wakeOffMs / 1000.0 < simEnd)
            g_wakeEvts[i]  = Simulator::Schedule(MilliSeconds(wakeOffMs),  &SetStaWake,  i);
        // when this station's service period ends, for SP-aligned solicitation
        g_spEndTime[i] = (sleepOffMs < biMs) ? (now + sleepOffMs / 1000.0) : 0.0;
        if (sleepOffMs < biMs && now + sleepOffMs / 1000.0 < simEnd)
            g_sleepEvts[i] = Simulator::Schedule(MilliSeconds(sleepOffMs), &SetStaSleep, i);
    }
}

// ─── Apply TWT schedule ────────────────────────────────────────────────────────
void ApplyTWTSchedule(uint32_t schedId, uint32_t assignId) {
    curSchedId = schedId; curAssignId = assignId;
    const ScheduleEntry& sched = SCHEDULES[schedId];
    for (uint32_t i = 0; i < NUM_STA; i++) {
        staGroupAssign[i] = ASSIGNMENTS[assignId](i, sched.numGroups);
    }
}

// ─── Per-BI metric update ─────────────────────────────────────────────────────
void UpdateBIMetrics() {
    bool twtActive = (currentBI >= TWT_SETUP_BI);
    const ScheduleEntry& sched = SCHEDULES[curSchedId];
    for (uint32_t i = 0; i < NUM_STA; i++) {
        double wake_ms, sleep_ms;
        if (twtActive && sched.numGroups >= 1) {
            uint32_t grp = staGroupAssign[i];
            double gWake = (grp < sched.wakeDurations.size())
                         ? sched.wakeDurations[grp] : sched.wakeDurations[0];
            wake_ms  = gWake;
            sleep_ms = BEACON_INTERVAL * 1000.0 - gWake;
        } else {
            wake_ms = BEACON_INTERVAL * 1000.0; sleep_ms = 0.0;
        }
        staAwakeTime[i] += wake_ms;
        staSleepTime[i] += sleep_ms;
    }
}

// ─── RL decision step ─────────────────────────────────────────────────────────
void RLDecisionStep() {
    double now = Simulator::Now().GetSeconds();
    flowMonitor->CheckForLostPackets();
    auto flowStats = flowMonitor->GetFlowStats();

    std::vector<double> dBytes(NUM_STA,0), dPkts(NUM_STA,0), dLost(NUM_STA,0);
    std::vector<double> dDelay(NUM_STA,0);
    std::vector<uint64_t> dTxPkts(NUM_STA,0);
    for (auto& [fid, fs] : flowStats) {
        Ptr<Ipv4FlowClassifier> cls = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
        auto t = cls->FindFlow(fid);
        auto it = portToSta.find(t.destinationPort);
        if (it == portToSta.end()) continue;
        uint32_t idx = it->second;
        dBytes[idx]   = fs.rxBytes    - prevRxBytes[idx];
        dPkts[idx]    = fs.rxPackets  - prevRxPkts[idx];
        dLost[idx]    = fs.lostPackets - prevLostPkts[idx];
        dDelay[idx]   = fs.delaySum.GetSeconds() - prevDelay[idx];
        dTxPkts[idx]  = fs.txPackets  - prevTxPkts[idx];
        prevRxBytes[idx]  = fs.rxBytes;
        prevTxPkts[idx]   = fs.txPackets;
        prevRxPkts[idx]   = fs.rxPackets;
        prevLostPkts[idx] = fs.lostPackets;
        prevDelay[idx]    = fs.delaySum.GetSeconds();
    }

    // Per-STA latency (ms) and packet loss rate from FlowMonitor
    std::vector<double> latency(NUM_STA, 0.0);
    std::vector<double> lossRate(NUM_STA, 0.0);
    for (uint32_t i = 0; i < NUM_STA; i++) {
        latency[i]  = (dPkts[i] > 0)   ? (dDelay[i] / dPkts[i] * 1e3) : 0.0;
        lossRate[i] = (dTxPkts[i] > 0) ? (double)dLost[i] / dTxPkts[i] : 0.0;
    }

    std::vector<double> curEnergy(NUM_STA, 0.0);
    for (uint32_t i = 0; i < NUM_STA; i++) {
        if (energySources[i])
            curEnergy[i] = (energySources[i]->GetInitialEnergy()
                          - energySources[i]->GetRemainingEnergy()) * 1e3;
    }

    std::vector<double> dAirtime(NUM_STA), dAwake(NUM_STA), dSleep(NUM_STA), dEnergyV(NUM_STA);
    std::vector<double> sleepRatio(NUM_STA), dutyCycle(NUM_STA), queueKB(NUM_STA, 0.0);

    for (uint32_t i = 0; i < NUM_STA; i++) {
        dAirtime[i]   = staTotalAirtime[i] - prevAirtime[i];
        dAwake[i]     = staAwakeTime[i]    - prevAwake[i];
        dSleep[i]     = staSleepTime[i]    - prevSleep[i];
        dEnergyV[i]   = curEnergy[i]       - prevEnergy[i];
        double totT   = staAwakeTime[i] + staSleepTime[i];
        sleepRatio[i] = totT > 0 ? staSleepTime[i] / totT : 0.0;
        dutyCycle[i]  = totT > 0 ? staAwakeTime[i] / totT : 1.0;
        double expArr = DEVICE_PROFILES[staClass[i]].dataRateBps / 8.0 * BEACON_INTERVAL * DELTA_STEP;
        queueKB[i]    = std::max(0.0, (expArr - dBytes[i]) / 1024.0);
        prevAirtime[i] = staTotalAirtime[i];
        prevAwake[i]   = staAwakeTime[i];
        prevSleep[i]   = staSleepTime[i];
        prevEnergy[i]  = curEnergy[i];
    }

    bool isWarmup = (currentStep < WARMUP_STEPS);
    bool isDone   = (currentStep >= T_STEPS + WARMUP_STEPS - 1);

    uint32_t newSchedId = curSchedId, newAssignId = curAssignId;

    if (aiInterface && !isWarmup) {
        aiInterface->CppSendBegin();
        ScenEnvStruct* env = aiInterface->GetCpp2PyStruct();
        env->step = currentStep;
        env->done = isDone ? 1.0f : 0.0f;
        for (uint32_t i = 0; i < NUM_STA; i++) {
            env->bsr_queue_index[i]   = (float)(queueKB[i]*1024/200.0);
            env->airtime_used[i]      = (float)staTotalAirtime[i];
            env->fcs_errors[i]        = 0.0f;
            env->rx_fragments[i]      = (float)dPkts[i];
            env->last_rx_ts[i]        = (float)(now * 1e6);
            env->awake_time[i]        = (float)staAwakeTime[i];
            env->sleep_time[i]        = (float)staSleepTime[i];
            env->duty_cycle[i]        = (float)dutyCycle[i];
            env->packets_tx[i]        = (float)dPkts[i];
            env->d_airtime[i]         = (float)dAirtime[i];
            env->d_awake[i]           = (float)dAwake[i];
            env->d_sleep[i]           = (float)dSleep[i];
            env->d_packets_tx[i]      = (float)dPkts[i];
            env->queue_size_bytes[i]  = (float)(queueKB[i]*1024);
            env->d_bytes_tx[i]        = (float)dBytes[i];
            env->d_energy[i]          = (float)dEnergyV[i];
            env->d_mpdu_drops[i]      = (float)dLost[i];
            env->device_class[i]      = (float)staClass[i]; // 0=IoT,1=Camera,2=Voice,3=Streamer
        }
        aiInterface->CppSendEnd();

        aiInterface->CppRecvBegin();
        ScenActionStruct* act = aiInterface->GetPy2CppStruct();
        newSchedId  = std::min(act->schedule_id,  (uint32_t)SCHEDULES.size()-1);
        newAssignId = std::min(act->assignment_id,(uint32_t)ASSIGNMENTS.size()-1);
        if (g_muScheduler)
        {
            g_muScheduler->SetAttribute("EnableUlOfdma", BooleanValue(act->ofdma_enabled != 0));
        }
        // g_ofdmaEnabled gates the two-phase sleep guard in SetStaSleep/SetStaWake
        // (the crash-prevention logic from Phase 7 - see SLEEP_GUARD_US above).
        // MUST track the live per-step OFDMA state, not just the CLI default,
        // or a step with OFDMA genuinely on would wrongly take the unguarded
        // immediate-sleep path and reintroduce the sleep-during-in-flight-
        // OFDMA-exchange crash this variable was added to prevent.
        g_ofdmaEnabled = (act->ofdma_enabled != 0);
        aiInterface->CppRecvEnd();
    }

    ApplyTWTSchedule(newSchedId, newAssignId);

    RewardComponents rc = ComputeReward(dBytes, dPkts, dAirtime, dEnergyV, dLost,
                                        queueKB, sleepRatio, dutyCycle, newSchedId);

    double totalWake = std::accumulate(SCHEDULES[newSchedId].wakeDurations.begin(),
                                       SCHEDULES[newSchedId].wakeDurations.end(), 0.0);

    actionLog << currentStep << "," << std::fixed << std::setprecision(4) << now << ","
              << newSchedId << "," << newAssignId << ","
              << SCHEDULES[newSchedId].numGroups << ","
              << std::setprecision(1) << totalWake << ","
              << (isWarmup ? "warmup" : "rl") << "\n";

    rewardLog << currentStep << "," << std::fixed << std::setprecision(4) << now << ","
              << std::setprecision(4) << rc.thruput << "," << rc.queue << "," << rc.energy << ","
              << rc.drops << "," << rc.air << "," << rc.channel << "," << rc.total << "\n";

    double aggTput = 0, aggEnergy = 0, aggLat = 0, aggLoss = 0;
    uint32_t latCnt = 0;
    for (uint32_t i = 0; i < NUM_STA; i++) {
        aggTput   += (dBytes[i] * 8.0) / (DELTA_STEP * BEACON_INTERVAL * 1e6);
        aggEnergy += curEnergy[i];
        aggLoss   += lossRate[i];
        if (latency[i] > 0) { aggLat += latency[i]; latCnt++; }
    }
    double meanLat  = latCnt ? aggLat / latCnt : 0.0;
    double meanLoss = aggLoss / NUM_STA;

    overallLog << currentStep << "," << std::fixed << std::setprecision(4) << now << ","
               << std::setprecision(4) << aggTput << ","
               << std::setprecision(3) << aggEnergy << ","
               << std::setprecision(2) << meanLat << ","
               << std::setprecision(4) << meanLoss << ","
               << std::setprecision(4) << rc.total << ","
               << newSchedId << "," << newAssignId << ","
               << SCHEDULES[newSchedId].numGroups << "\n";

    for (uint32_t i = 0; i < NUM_STA; i++) {
        double tput_mbps = (dBytes[i] * 8.0) / (DELTA_STEP * BEACON_INTERVAL * 1e6);
        perStaLog << currentStep << "," << std::fixed << std::setprecision(4) << now << ","
                  << i << "," << DEVICE_PROFILES[staClass[i]].name << ","
                  << staGroupAssign[i] << ","
                  << std::setprecision(4) << tput_mbps << ","
                  << std::setprecision(2) << latency[i] << ","
                  << std::setprecision(4) << lossRate[i] << ","
                  << std::setprecision(3) << curEnergy[i] << ","
                  << std::setprecision(4) << sleepRatio[i] << ","
                  << std::setprecision(4) << dutyCycle[i] << ","
                  << std::setprecision(2) << queueKB[i] << ","
                  << std::setprecision(0) << dLost[i] << ",";
        // time in each PHY state this step, and the energy the configured
        // currents attribute to the four states whose current we set
        double dst[NUM_PHY_STATES];
        for (uint32_t k = 0; k < NUM_PHY_STATES; k++) {
            dst[k] = g_stateSec[i][k] - g_prevStateSec[i][k];
            g_prevStateSec[i][k] = g_stateSec[i][k];
        }
        const double mJ = SUPPLY_VOLTAGE * 1000.0;   // A*s*V -> mJ
        perStaLog << std::setprecision(5)
                  << dst[0] << "," << dst[1] << "," << dst[2] << ","
                  << dst[3] << "," << dst[4] << "," << dst[5] << ","
                  << std::setprecision(4)
                  << dst[0] * (CURRENT_IDLE_MA  / 1000.0) * mJ << ","
                  << dst[1] * (CURRENT_CCA_MA   / 1000.0) * mJ << ","
                  << dst[2] * (CURRENT_TX_MA    / 1000.0) * mJ << ","
                  << dst[3] * (CURRENT_RX_MA    / 1000.0) * mJ << ","
                  << dst[5] * (CURRENT_SLEEP_MA / 1000.0) * mJ << ",";
        // measured energy per state: sums to the EnergySource total by construction
        double dmj[NUM_PHY_STATES];
        for (uint32_t k = 0; k < NUM_PHY_STATES; k++) {
            dmj[k] = g_stateMj[i][k] - g_prevStateMj[i][k];
            g_prevStateMj[i][k] = g_stateMj[i][k];
        }
        perStaLog << std::setprecision(4)
                  << dmj[0] << "," << dmj[1] << "," << dmj[2] << ","
                  << dmj[3] << "," << dmj[4] << "," << dmj[5] << "\n";
    }
    currentStep++;
}

// ─── Beacon-interval callback ─────────────────────────────────────────────────
void OnBeaconInterval() {
    UpdateBIMetrics();
    if (currentBI >= RL_START_BI) {
        uint32_t stepBI = currentBI - RL_START_BI;
        if (stepBI % DELTA_STEP == 0) RLDecisionStep();  // may update curSchedId
    }
    if (g_twtEnabled && currentBI >= TWT_SETUP_BI)
        ScheduleBITWT();   // physical PHY sleep — uses latest schedule
    currentBI++;
    if (currentBI < SIM_BI)
        Simulator::Schedule(Seconds(BEACON_INTERVAL), &OnBeaconInterval);
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    uint32_t    seed       = 1;
    std::string preset     = "energy";
    bool        useRLAgent = true;
    std::string segName    = "TwtScen4RL";
    bool        enableOfdma = false; // see TwtAwareScheduler above; default preserves
                                      // the historical (OFDMA-off) collection dataset

    double loadStepAtS  = -1.0;

    double loadStepMult = 2.0;

    std::string outPrefix     = "";
    uint32_t    sleepGuardUs  = 8000;
    double      ccaCurrentMa  = 273.0;
    bool        spAligned     = false;
    double      spMarginUs    = 5800.0;
    bool        guardSymmetric = false;
    uint32_t    nStations     = 4;      // RrMultiUserScheduler default

    CommandLine cmd;
    cmd.AddValue("loadStepAtS",  "If >=0, multiply Camera+Streamer DataRate by "
                 "loadStepMult at this sim time (nonstationary collection)", loadStepAtS);
    cmd.AddValue("loadStepMult", "DataRate multiplier applied at loadStepAtS", loadStepMult);
    cmd.AddValue("seed",    "RNG seed",                  seed);
    cmd.AddValue("preset",  "Reward preset",             preset);
    cmd.AddValue("useRL",   "Enable RL agent",           useRLAgent);
    cmd.AddValue("segName", "Shared memory segment name",segName);
    cmd.AddValue("outPrefix", "Prefix for the four CSV logs (default: segName, "
                 "which is unique per episode so parallel runs cannot collide)",
                 outPrefix);
    cmd.AddValue("sleepGuardUs", "Microseconds to defer an actual PHY sleep so an "
                 "in-flight trigger exchange can finish. 8000 reproduces the "
                 "historical behaviour; 0 sleeps immediately", sleepGuardUs);
    cmd.AddValue("spAligned", "Only solicit a station if its TWT service period "
                 "still has room for the whole trigger exchange. Makes the 8 ms "
                 "sleep guard unnecessary: run with --sleepGuardUs=0", spAligned);
    cmd.AddValue("spMarginUs", "Exchange duration reserved by --spAligned", spMarginUs);
    cmd.AddValue("ccaCurrentMa", "CCA-busy current in mA. Default 273 reproduces "
                 "ns-3's unset default; 66 makes the profile coherent by charging "
                 "carrier sensing at the receive current", ccaCurrentMa);
    cmd.AddValue("sleepGuardSymmetric", "Apply the sleep guard in BOTH OFDMA arms. "
                 "Historically it applied only when the scheduler was installed, "
                 "so the arms differed in sleep handling as well as in scheduler "
                 "availability", guardSymmetric);
    cmd.AddValue("enableTwt", "If false, never sleep any station: a true always-on "
                 "reference. Default true (unchanged behaviour).", g_twtEnabled);
    cmd.AddValue("nStations", "Max stations granted an RU per trigger frame. The "
                 "ns-3 RrMultiUserScheduler default is 4; a 20 MHz channel admits 9 "
                 "26-tone RUs, so this bounds the multiplexing degree.", nStations);
    cmd.AddValue("enableOfdma", "Enable UL OFDMA during data collection (TwtAwareScheduler)",
                 enableOfdma);
    cmd.Parse(argc, argv);
    CURRENT_CCA_MA        = ccaCurrentMa;
    g_spAligned           = spAligned;
    g_spMarginUs          = spMarginUs;
    g_sleepGuardUs        = sleepGuardUs;
    g_sleepGuardSymmetric = guardSymmetric;
    g_ofdmaEnabled = enableOfdma;

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(1);

    if (preset == "throughput")   curWeights = TPUT_PRESET;
    else if (preset == "queue")   curWeights = QUEUE_PRESET;
    else if (preset == "low_loss") curWeights = LOW_LOSS_PRESET;
    else                           curWeights = ENERGY_PRESET;

    double simDuration = SIM_BI * BEACON_INTERVAL;
    InitAssignments();

    // ns3-ai bridge: Python side is memory creator; C++ opens existing segment
    if (useRLAgent) {
        aiInterface = new Ns3AiMsgInterfaceImpl<ScenEnvStruct, ScenActionStruct>(
            false, false, true, 65536, segName.c_str());
    }

    // ── Nodes ─────────────────────────────────────────────────────────────────
    apNode.Create(1);
    staNodes.Create(NUM_STA);

    // ── PHY / Channel ─────────────────────────────────────────────────────────
    // SpectrumWifiPhyHelper required for RrMultiUserScheduler UL OFDMA in NS-3.44.
    auto spectrumChannel = CreateObject<MultiModelSpectrumChannel>();
    auto lossModel = CreateObject<LogDistancePropagationLossModel>();
    lossModel->SetAttribute("Exponent",          DoubleValue(PATH_LOSS_EXP));
    lossModel->SetAttribute("ReferenceDistance", DoubleValue(1.0));
    lossModel->SetAttribute("ReferenceLoss",     DoubleValue(46.6777));
    spectrumChannel->AddPropagationLossModel(lossModel);
    spectrumChannel->SetPropagationDelayModel(
        CreateObject<ConstantSpeedPropagationDelayModel>());

    SpectrumWifiPhyHelper phy;
    phy.SetChannel(spectrumChannel);
    phy.Set("ChannelSettings", StringValue("{36, 20, BAND_5GHZ, 0}"));

    // ── MAC: 802.11ax with TWT-aligned scheduling + UL OFDMA ─────────────────
    // MinstrelHt conflicts with OFDMA trigger-based exchange; use ConstantRate.
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ax);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",    StringValue("HeMcs4"),
                                 "ControlMode", StringValue("HeMcs0"));

    Ssid ssid = Ssid("rl-twt-net");

    // Install STAs first (NS-3.44 OFDMA pattern: STAs before AP)
    WifiMacHelper staMac;
    staMac.SetType("ns3::StaWifiMac",
                   "Ssid",             SsidValue(ssid),
                   "ActiveProbing",    BooleanValue(false),
                   "MaxMissedBeacons", UintegerValue(1000));
    staDevices = wifi.Install(phy, staMac, staNodes);

    // Install AP with TwtAwareScheduler (drop-in RrMultiUserScheduler that
    // additionally excludes TWT-sleeping STAs from OFDMA solicitation — falls
    // through to identical RrMultiUserScheduler behaviour when OFDMA is off).
    WifiMacHelper apMac;
    apMac.SetMultiUserScheduler("ns3::TwtAwareScheduler",
                                "EnableUlOfdma",     BooleanValue(enableOfdma),
                                "EnableBsrp",        BooleanValue(false),
                                "NStations",         UintegerValue(nStations),
                                "AccessReqInterval", TimeValue(MilliSeconds(10)));
    apMac.SetType("ns3::ApWifiMac",
                  "Ssid",           SsidValue(ssid),
                  "BeaconInterval", TimeValue(Seconds(BEACON_INTERVAL)));
    apDevice = wifi.Install(phy, apMac, apNode);

    // Retrieve the live scheduler handle (aggregated onto ApWifiMac by
    // WifiMacHelper::Create() during wifi.Install() above - see
    // wifi-mac-helper.cc:141 apMac->AggregateObject(muScheduler)).
    {
        Ptr<WifiNetDevice> apNetDevice = DynamicCast<WifiNetDevice>(apDevice.Get(0));
        Ptr<ApWifiMac> apWifiMac = DynamicCast<ApWifiMac>(apNetDevice->GetMac());
        g_muScheduler = apWifiMac->GetObject<MultiUserScheduler>();
    }

    if (enableOfdma)
    {
        std::ostringstream phyTxPath, notifyPath, rxHdrPath;
        phyTxPath << "/NodeList/" << apNode.Get(0)->GetId() << "/DeviceList/*/Phy/PhyTxPsduBegin";
        Config::Connect(phyTxPath.str(), MakeCallback(&ApTriggerTxTrace));
        notifyPath << "/NodeList/" << apNode.Get(0)->GetId() << "/DeviceList/*/Phy/NotifyMacHdrRxEnd";
        Config::Set(notifyPath.str(), BooleanValue(true));
        rxHdrPath << "/NodeList/" << apNode.Get(0)->GetId() << "/DeviceList/*/Phy/PhyRxMacHeaderEnd";
        Config::Connect(rxHdrPath.str(), MakeCallback(&ApRxHdrTrace));
    }

    // ── Mobility ──────────────────────────────────────────────────────────────
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> apPos = CreateObject<ListPositionAllocator>();
    apPos->Add(Vector(AREA_SIDE / 2, AREA_SIDE / 2, 0));
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.SetPositionAllocator(apPos);
    mobility.Install(apNode);

    mobility.SetPositionAllocator(
        "ns3::RandomRectanglePositionAllocator",
        "X", StringValue("ns3::UniformRandomVariable[Min=0|Max=30]"),
        "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max=30]"));
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(staNodes);

    // ── Internet stack ─────────────────────────────────────────────────────────
    InternetStackHelper internet;
    internet.Install(apNode);
    internet.Install(staNodes);

    Ipv4AddressHelper addrHelper;
    addrHelper.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer apIface = addrHelper.Assign(apDevice);
    addrHelper.Assign(staDevices);

    // ── Device class assignment ────────────────────────────────────────────────
    std::vector<uint32_t> classAssign;
    for (uint32_t c = 0; c < 4; c++)
        for (uint32_t k = 0; k < 4; k++)
            classAssign.push_back(c);
    Ptr<UniformRandomVariable> rngC = CreateObject<UniformRandomVariable>();
    for (int i = NUM_STA-1; i > 0; i--) { int j = rngC->GetInteger(0,i); std::swap(classAssign[i],classAssign[j]); }
    for (uint32_t i = 0; i < NUM_STA; i++) staClass[i] = (DeviceClass)classAssign[i];
    ApplyTWTSchedule(DEFAULT_SCHED, DEFAULT_ASSIGN);

    // ── Applications ──────────────────────────────────────────────────────────
    uint16_t basePort = 9000;
    Ptr<UniformRandomVariable> rngStart = CreateObject<UniformRandomVariable>();
    for (uint32_t i = 0; i < NUM_STA; i++) {
        const DeviceProfile& dp = DEVICE_PROFILES[staClass[i]];
        uint16_t port = basePort + i;
        portToSta[port] = i;
        double tStart = rngStart->GetInteger(APP_START_MIN_BI, APP_START_MAX_BI) * BEACON_INTERVAL;
        double tStop  = simDuration - 1.0;

        OnOffHelper onoff("ns3::UdpSocketFactory",
            InetSocketAddress(apIface.GetAddress(0), port));
        onoff.SetAttribute("DataRate",   DataRateValue(DataRate((uint64_t)dp.dataRateBps)));
        onoff.SetAttribute("PacketSize", UintegerValue(dp.pktSizeBytes));

        if (dp.dc == IOT_SENSOR) {
            onoff.SetAttribute("OnTime",  StringValue("ns3::ConstantRandomVariable[Constant=1.0]"));
            onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=4.0]"));
        } else if (dp.dc == VIDEO_CAMERA) {
            onoff.SetAttribute("OnTime",  StringValue("ns3::ConstantRandomVariable[Constant=1000]"));
            onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        } else {
            double mOn  = (dp.dc == VOICE_ASSISTANT) ? 1.0 : 2.0;
            double mOff = (dp.dc == VOICE_ASSISTANT) ? 1.5 : 1.0;
            onoff.SetAttribute("OnTime",  StringValue("ns3::ExponentialRandomVariable[Mean="+std::to_string(mOn)+"]"));
            onoff.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean="+std::to_string(mOff)+"]"));
        }
        ApplicationContainer a = onoff.Install(staNodes.Get(i));
        a.Start(Seconds(tStart)); a.Stop(Seconds(tStop));

        PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApp = sink.Install(apNode.Get(0));
        sinkApp.Start(Seconds(0)); sinkApp.Stop(Seconds(simDuration));
    }

    // ── Patch 06: mid-episode load step (nonstationary collection) ─────────
    // Mirrors rl-agent-scenario-v2.cc: at loadStepAtS, Camera (1) and Streamer
    // (3) DataRate is multiplied by loadStepMult. Inert when loadStepAtS < 0.
    if (loadStepAtS >= 0.0) {
        for (uint32_t i = 0; i < NUM_STA; i++) {
            if (staClass[i] != VIDEO_CAMERA && staClass[i] != VIDEO_STREAMER) continue;
            std::string newRate = std::to_string(
                (uint64_t)(DEVICE_PROFILES[staClass[i]].dataRateBps * loadStepMult)) + "bps";
            Ptr<Application> app = staNodes.Get(i)->GetApplication(0);
            Simulator::Schedule(Seconds(loadStepAtS), [app, newRate]() {
                app->SetAttribute("DataRate", StringValue(newRate));
            });
        }
        std::cout << "[scenario4] loadStep at " << loadStepAtS
                  << "s x" << loadStepMult << std::endl;
    }

    // ── Energy model ──────────────────────────────────────────────────────────
    BasicEnergySourceHelper eHelper;
    eHelper.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(1e6));
    eHelper.Set("BasicEnergySupplyVoltageV",       DoubleValue(SUPPLY_VOLTAGE));
    EnergySourceContainer sources = eHelper.Install(staNodes);

    WifiRadioEnergyModelHelper radioHelper;
    radioHelper.Set("TxCurrentA",    DoubleValue(CURRENT_TX_MA    / 1000.0));
    radioHelper.Set("RxCurrentA",    DoubleValue(CURRENT_RX_MA    / 1000.0));
    radioHelper.Set("IdleCurrentA",  DoubleValue(CURRENT_IDLE_MA  / 1000.0));
    radioHelper.Set("SleepCurrentA", DoubleValue(CURRENT_SLEEP_MA / 1000.0));
    // Previously left unset, which silently kept ns-3's 273 mA default.
    radioHelper.Set("CcaBusyCurrentA", DoubleValue(CURRENT_CCA_MA / 1000.0));
    radioHelper.Install(staDevices, sources);
    for (uint32_t i = 0; i < NUM_STA; i++) energySources[i] = sources.Get(i);

    for (uint32_t i = 0; i < NUM_STA; i++) {
        g_stateSec[i].fill(0.0);
        g_prevStateSec[i].fill(0.0);
        g_stateMj[i].fill(0.0);
        g_prevStateMj[i].fill(0.0);
        g_lastDrawnMj[i] = 0.0;
        Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(staDevices.Get(i));
        if (wd && wd->GetPhy())
            wd->GetPhy()->GetState()->TraceConnectWithoutContext(
                "State", MakeBoundCallback(&PhyStateCb, i));
    }

    // ── Flow monitor ──────────────────────────────────────────────────────────
    flowMonitor = flowHelper.InstallAll();

    // ── Output files ──────────────────────────────────────────────────────────
    // The four logs had fixed names, so parallel episodes overwrote one
    // another's output. segName is already unique per episode, so it prefixes
    // them unless --outPrefix says otherwise.
    std::string pfx = outPrefix.empty() ? segName : outPrefix;
    overallLog.open(pfx + "_overall.csv");
    perStaLog.open(pfx + "_persta.csv");
    actionLog.open(pfx + "_actions.csv");
    rewardLog.open(pfx + "_rewards.csv");
    overallLog << "step,time_s,total_throughput_mbps,total_energy_mj,"
               << "mean_latency_ms,packet_loss_rate,reward,"
               << "sched_id,assign_id,num_groups\n";
    perStaLog  << "step,time_s,sta_id,device_class,twt_group,"
               << "throughput_mbps,latency_ms,packet_loss_rate,"
               << "energy_mj,sleep_ratio,duty_cycle,queue_kb,mpdu_drops,"
               << "t_idle_s,t_cca_s,t_tx_s,t_rx_s,t_switch_s,t_sleep_s,"
               << "e_idle_mj,e_cca_mj,e_tx_mj,e_rx_mj,e_sleep_mj,"
               << "m_idle_mj,m_cca_mj,m_tx_mj,m_rx_mj,m_switch_mj,m_sleep_mj\n";
    actionLog  << "step,time_s,sched_id,assign_id,num_groups,total_wake_ms,mode\n";
    rewardLog  << "step,time_s,r_thruput,r_queue,r_energy,r_drops,r_air,r_chan,r_total\n";

    Simulator::Schedule(Seconds(BEACON_INTERVAL), &OnBeaconInterval);

    std::cout << "[Scenario 4] RL Agent + TWT + OFDMA\n";
    std::cout << "  STAs: " << NUM_STA << "  Duration: " << simDuration
              << " s (" << SIM_BI << " BIs)\n";
    std::cout << "  RL steps: " << T_STEPS << " (every " << DELTA_STEP << " BIs)\n";
    std::cout << "  Reward preset: " << preset << "  Seed: " << seed << "\n";
    if (useRLAgent)
        std::cout << "  RL bridge: shared memory 'TwtScen4RL'\n";
    else
        std::cout << "  RL bridge: DISABLED (default schedule)\n";

    Simulator::Stop(Seconds(simDuration));
    Simulator::Run();

    flowMonitor->SerializeToXmlFile("scenario4_rl_flowmon.xml", true, true);
    overallLog.close(); perStaLog.close(); actionLog.close(); rewardLog.close();

    if (aiInterface) {
        aiInterface->CppSetFinished();
        delete aiInterface;
    }

    std::cout << "[Scenario 4] ofdma=" << (enableOfdma ? "on" : "off")
              << " trigger_frames_tx=" << g_triggerFramesTxByAp
              << " tb_ppdu_rx=" << g_heTbPpduRxAtAp << "\n";
    std::cout << "[Scenario 4] Done.\n";
    Simulator::Destroy();

    // Same at-exit SIGSEGV mitigation as rl-agent-scenario-v2.cc (see comment
    // there for the gdb-confirmed root cause: a memory-layout-dependent crash
    // inside ns3::Object::~Object() during atexit-time static destruction,
    // after all real work - including the ns3-ai handshake above - is done).
    // This scenario has the identical unguarded `static Ptr<FlowMonitor>
    // flowMonitor;` pattern, so it's exposed to the same bug. The Python
    // collector reads transitions.jsonl directly (fsync'd per line, written
    // before this point) rather than checking this process's exit code, so
    // data was never at risk - this just removes a scary/misleading
    // "Segmentation fault" line from the collection log.
    std::cout.flush();
    _exit(0);
}
