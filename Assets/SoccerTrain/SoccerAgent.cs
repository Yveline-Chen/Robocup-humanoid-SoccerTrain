using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class SoccerAgent : Agent
{
    static SoccerAgent navProbeAgent;
    [Header("Robot Config")]
    public bool fixbody = false;
    public bool train;
    public Unity.Sentis.ModelAsset policyModel;

    [Header("Inference Diagnostics")]
    [SerializeField] bool debugPerEnvironment = true;
    [SerializeField] bool debugNavigation = false;
    const int envDebugWindow = 10;
    int envDebugEpisodes;
    int envDebugWindowIndex;
    int envDebugGoals;
    int envDebugFalls;
    int envDebugNoMove;
    int envDebugDecisionCalls;
    int envDebugActionValues;
    int envDebugSpeedSamples;
    double envDebugLifeSum;
    double envDebugTravelSum;
    double envDebugSpeedSum;
    double envDebugActionSum;
    float envEpisodeTravel;
    int inferenceEpisodeIndex;
    int lastNavProbeState = -1;
    int lastNavProbeStep = -1000;
    string diagnosticEnvId;

    [Header("Soccer Curriculum")]
    static int globalPhase = 3;
    public float phase2Threshold = 90f;
    public float phase3Threshold = 160f;
    [Range(5, 50)] public int phaseWindow = 20;

    float[] recentRewards;
    int localPhase;
    int rwIdx, rwCount;

    ArticulationBody[] allBodies, revoluteJoints;
    int numJoints;
    ArticulationBody rootBody;
    float initY;
    Vector3 spawnPos;
    Vector3 goalDir;

    // 12 leg joints
    int[] legIdx = new int[12];

    Transform ball; Rigidbody ballRb;
    Vector3 goalLeftPos;
    float fieldHalfL = 7f, fieldHalfW = 4.5f;
    const float BALL_R = 0.11f;

    // variables for gait control
    int epStep;
    int tp = 0;
    int tq = 0;
    int T1 = 40;
    int T2 = 40;
    int tp0 = 0;

    float uf1 = 0; // left leg
    float uf2 = 0; // right leg
    float uff = 0; // kick motion

    bool isKicking = false;
    bool kickBalanceMaintained = false;
    bool kickCompletionPending = false;
    bool stableKickRewarded = false;
    int kickWaitTimer = 0;
    bool isStrikingPhase = false;
    int strikeLockFrames = 0;
    int strikeExitFrames = 0;
    bool strikeRetryUsed = false;
    Vector3 strikeHeading = Vector3.zero;
    bool followBallActive = false;
    Vector3 previousBallVelocity = Vector3.zero;
    int shotEvaluationDelay = 0;
    int shotContactCooldown = 0;
    bool pendingShotWasKick = false;

    // Phase 3 Metrics
    static int p3WindowEpisodes = 0;
    static int p3DebugWindowIndex = 0;
    static int p3WindowGoals = 0;
    static int p3WindowShotEpisodes = 0;
    static int p3WindowAccurateFirstShots = 0;
    static int p3WindowKickEpisodes = 0;
    static int p3WindowFalls = 0;
    static int p3FallApproach = 0;
    static int p3FallAlign = 0;
    static int p3FallStrike = 0;
    static int p3FallKick = 0;
    static int p3FallFollow = 0;
    static int p3FirstContactEpisodes = 0;
    static int p3FirstContactApproach = 0;
    static int p3FirstContactAlign = 0;
    static int p3FirstContactStrike = 0;
    static int p3FirstContactKick = 0;
    static int p3FirstContactFollow = 0;
    static int p3FirstContactRight = 0;
    static int p3FirstContactLeft = 0;
    static int p3FirstContactBoth = 0;
    static int p3FirstContactBodyAligned = 0;
    static int p3FirstContactToeAligned = 0;
    static int p3FirstContactRightDominant = 0;
    static int p3FirstContactBallMoving = 0;
    static int p3AvoidEnterEpisodes = 0;
    static int p3AvoidExitEpisodes = 0;
    static int p3AlignEnterEpisodes = 0;
    static int p3AlignExitEpisodes = 0;
    static int p3AlignReenterEpisodes = 0;
    static int p3AccurateApproach = 0;
    static int p3AccurateAlign = 0;
    static int p3AccurateStrike = 0;
    static int p3AccurateKick = 0;
    static int p3AccurateFollow = 0;
    static int p3AccurateRight = 0;
    static int p3AccurateLeft = 0;
    static int p3AccurateBoth = 0;
    static int p3AccurateRightDominant = 0;
    static int p3KickStarts = 0;
    static int p3KickStableCompletions = 0;
    static int p3KickContacts = 0;
    static double p3GoalTimeSum = 0.0;
    static int p3GoalTimeCount = 0;
    static double p3FallSpeedSum = 0.0;
    static double p3FallTiltSum = 0.0;
    static double p3FallResidualSum = 0.0;
    static int p3FallDynamicsSamples = 0;
    static double p3ResidualSum = 0.0;
    static long p3ResidualSamples = 0;
    static string debugRunId = "";
    static bool debugRunActive = false;
    static int debugRunAgents = 0;
    static int debugRunMaxAgents = 0;
    static int debugRunEpisodes = 0;
    static int debugRunGoals = 0;
    static int debugRunFalls = 0;
    static int debugRunKicks = 0;
    static int debugRunContacts = 0;
    bool everKicked = false;
    bool episodeScored = false;
    bool episodeFirstShotEvaluated = false;
    bool episodeFirstShotAccurate = false;
    bool episodeFell = false;
    bool episodeFirstContactLogged = false;
    int episodeFirstContactState = -1;
    int episodeFirstContactFoot = -1;
    bool episodeFirstContactBodyAligned = false;
    bool episodeFirstContactToeAligned = false;
    bool episodeFirstContactRightDominant = false;
    bool episodeFirstContactBallMoving = false;
    bool episodeKickContact = false;
    bool episodeAvoidEntered = false;
    bool episodeAvoidExited = false;
    bool episodeAlignEntered = false;
    bool episodeAlignExited = false;
    bool episodeAlignReentered = false;
    int episodeAlignEntries = 0;
    bool previousAvoiding = false;
    bool previousShootSpotLocked = false;
    int contactStateBeforeNavigation = 0;
    float currentResidualRms = 0f;

    bool shootSpotLocked = false;
    int shootSpotTimer = 0;
    int shootSpotStableFrames = 0;
    int alignExitFrames = 0;
    bool approachTargetLocked = false;
    Vector3 lockedApproachTarget = Vector3.zero;
    Vector3 lockedApproachBallPos = Vector3.zero;
    Vector3 lockedApproachDir = Vector3.zero;
    [SerializeField] Transform rightToeEdge;
    [SerializeField] private float rightToeHalfWidth = 0.025f;
    Vector3 previousRightToePosition = Vector3.zero;
    Vector3 rightToeVelocity = Vector3.zero;
    bool rightToeVelocityReady = false;

    float gaitKneeBias = 12f;
    float stanceHipBias = 6f;
    float[] gaitU, gaitTotal;

    float targetVel;
    float smoothedStrideScale = 1.0f;

    Vector3 targetMoveDir = Vector3.zero;
    Vector3 targetLookDir = Vector3.zero;
    float currentSpeedThreshold = 0.6f;

    readonly int[] sagJoints = { 0, 3, 4, 6, 9, 10 };
    readonly float[] sagSigns = { -1f, 1f, -1f, -1f, 1f, -1f };
    readonly float[] legGains = { 2000f, 3000f, 2500f, 2000f, 1000f, 500f, 2000f, 3000f, 2500f, 2000f, 1000f, 500f };
    readonly float[] legDamps = { 50f, 40f, 30f, 50f, 30f, 20f, 50f, 40f, 30f, 50f, 30f, 20f };
    readonly float[] legForces = { 150f, 100f, 80f, 150f, 60f, 40f, 150f, 100f, 80f, 150f, 60f, 40f };

    readonly string[] upN = { "pelvis_link", "waist_roll_link", "waist_yaw_link", "head_pitch_link", "head_yaw_link",
                              "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
                              "right_elbow_link", "right_wrist_roll_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
                              "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link" };

    readonly float[] us = { 2000f, 1500f, 1000f, 500f, 200f, 1000f, 800f, 800f, 800f, 300f, 1000f, 800f, 800f, 800f, 300f };
    readonly float[] uDamp = { 50f, 40f, 30f, 15f, 10f, 30f, 20f, 20f, 20f, 10f, 30f, 20f, 20f, 20f, 10f };
    readonly float[] uForce = { 150f, 150f, 100f, 50f, 20f, 80f, 60f, 60f, 60f, 20f, 80f, 60f, 60f, 60f, 20f };
    readonly float[] actScale = { 10f, 1.5f, 1.5f, 10f, 10f, 5f, 10f, 1.5f, 1.5f, 10f, 10f, 5f };

    float[] ud = new float[15];

    static int goalCount;
    bool hasTouchedBall;
    Transform leftFootTip, rightFootTip;
    float groundY;
    float maxSwingFootLift;

    float fieldLength;
    Vector3 fieldRight;

    [Header("Anti-Splay")]
    // dynamic yaw scaling
    public float yawScaleMin = 3f;
    public float yawScaleMax = 18f;     
    public float hipRollScaleMin = 2f;
    public float hipRollScaleMax = 8f;
    public float ankleRollScaleMin = 2f;
    public float ankleRollScaleMax = 8f;
    public float freedomRampSteps = 2000000f;
    public float rollDriftPenaltyCoef = 0.0001f;

    static double trainStepCounter = 0; 
    float hipRollEmaR, hipRollEmaL, ankleRollEmaR, ankleRollEmaL; 

    [Header("Phase3 Shootspot Shaping")]
    // intensive shootspot steering reward
    public float shootSpotProgressCoef = 0.5f;    
    public float shootSpotArrivalBonus = 0.5f;    
    float prevDistToShootSpot = -1f;
    bool reachedShootSpotOnce = false;

    void Start()
    {
        Time.fixedDeltaTime = 0.01f;
    }

    void StartDebugRun()
    {
        if (!debugRunActive)
        {
            debugRunId = System.DateTime.Now.ToString("HHmmssfff");
            debugRunActive = true;
            debugRunAgents = 0;
            debugRunMaxAgents = 0;
            debugRunEpisodes = 0;
            debugRunGoals = 0;
            debugRunFalls = 0;
            debugRunKicks = 0;
            debugRunContacts = 0;
            ResetP3Metrics(true);
            string modelName = policyModel != null ? policyModel.name : "NULL";
            Debug.Log($"[RUN START id={debugRunId}] model={modelName}");
        }
        debugRunAgents++;
        debugRunMaxAgents = Mathf.Max(debugRunMaxAgents, debugRunAgents);
    }

    protected override void OnDisable()
    {
        base.OnDisable();
        if (navProbeAgent == this) navProbeAgent = null;
        if (train || !debugRunActive) return;
        debugRunAgents = Mathf.Max(0, debugRunAgents - 1);
        if (debugRunAgents > 0) return;
        float goalRate = 100f * debugRunGoals / Mathf.Max(1, debugRunEpisodes);
        float fallRate = 100f * debugRunFalls / Mathf.Max(1, debugRunEpisodes);
        float kickRate = 100f * debugRunKicks / Mathf.Max(1, debugRunEpisodes);
        Debug.Log($"[RUN END id={debugRunId}] envs={debugRunMaxAgents} | episodes={debugRunEpisodes} | goals={debugRunGoals} | goalRate={goalRate:F1}% | falls={debugRunFalls} ({fallRate:F1}%) | isKick={debugRunKicks} ({kickRate:F1}%) | contacts={debugRunContacts}");
        debugRunActive = false;
    }

    static void ResetP3Metrics(bool resetWindowIndex)
    {
        if (resetWindowIndex) p3DebugWindowIndex = 0;
        p3WindowEpisodes = 0;
        p3WindowGoals = 0;
        p3WindowShotEpisodes = 0;
        p3WindowAccurateFirstShots = 0;
        p3WindowKickEpisodes = 0;
        p3WindowFalls = 0;
        p3FallApproach = 0;
        p3FallAlign = 0;
        p3FallStrike = 0;
        p3FallKick = 0;
        p3FallFollow = 0;
        p3FirstContactEpisodes = 0;
        p3FirstContactApproach = 0;
        p3FirstContactAlign = 0;
        p3FirstContactStrike = 0;
        p3FirstContactKick = 0;
        p3FirstContactFollow = 0;
        p3FirstContactRight = 0;
        p3FirstContactLeft = 0;
        p3FirstContactBoth = 0;
        p3FirstContactBodyAligned = 0;
        p3FirstContactToeAligned = 0;
        p3FirstContactRightDominant = 0;
        p3FirstContactBallMoving = 0;
        p3AvoidEnterEpisodes = 0;
        p3AvoidExitEpisodes = 0;
        p3AlignEnterEpisodes = 0;
        p3AlignExitEpisodes = 0;
        p3AlignReenterEpisodes = 0;
        p3AccurateApproach = 0;
        p3AccurateAlign = 0;
        p3AccurateStrike = 0;
        p3AccurateKick = 0;
        p3AccurateFollow = 0;
        p3AccurateRight = 0;
        p3AccurateLeft = 0;
        p3AccurateBoth = 0;
        p3AccurateRightDominant = 0;
        p3KickStarts = 0;
        p3KickStableCompletions = 0;
        p3KickContacts = 0;
        p3GoalTimeSum = 0.0;
        p3GoalTimeCount = 0;
        p3FallSpeedSum = 0.0;
        p3FallTiltSum = 0.0;
        p3FallResidualSum = 0.0;
        p3FallDynamicsSamples = 0;
        p3ResidualSum = 0.0;
        p3ResidualSamples = 0;
    }

    public override void Initialize()
    {
        if (!train) StartDebugRun();
        if (navProbeAgent == null) navProbeAgent = this;
        allBodies = GetComponentsInChildren<ArticulationBody>();
        var list = new System.Collections.Generic.List<ArticulationBody>();
        foreach (var ab in allBodies)
            if (ab.jointType == ArticulationJointType.RevoluteJoint)
                list.Add(ab);
        revoluteJoints = list.ToArray();
        numJoints = revoluteJoints.Length;
        gaitU = new float[numJoints];
        gaitTotal = new float[numJoints];

        string[] legNames = { "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
                              "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
                              "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
                              "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link" };
        for (int k = 0; k < 12; k++)
            legIdx[k] = -1;
        for (int i = 0; i < numJoints; i++)
            for (int k = 0; k < 12; k++)
                if (revoluteJoints[i].name == legNames[k])
                    legIdx[k] = i;

        rootBody = allBodies[0];
        initY = rootBody.transform.position.y;
        spawnPos = rootBody.transform.position;
        diagnosticEnvId = $"{(transform.parent != null ? transform.parent.name : name)}@{spawnPos.x:F1},{spawnPos.z:F1}";

        if (fixbody)
        {
            rootBody.immovable = true;
            initY += 0.3f;
            spawnPos.y += 0.3f;
        }

        var p = transform.parent;
        if (p != null)
        {
            var b = p.Find("SoccerBall");
            if (b)
            {
                ball = b;
                ballRb = b.GetComponent<Rigidbody>();
            }

            var g = FindChildRecursive(p, "Goal_Left");
            if (g)
            {
                goalLeftPos = g.position;
                Vector3 vecToGoal = goalLeftPos - spawnPos;
                vecToGoal.y = 0;
                if (vecToGoal == Vector3.zero) 
                    vecToGoal = Vector3.forward;
                goalDir = vecToGoal.normalized;
                fieldLength = vecToGoal.magnitude;
                fieldRight = Vector3.Cross(Vector3.up, goalDir).normalized;
            }
        }

        recentRewards = new float[phaseWindow];
        var rf = FindChildRecursive(transform, "right_ankle_roll_link");
        var lf = FindChildRecursive(transform, "left_ankle_roll_link");
        if (rf) 
            rightFootTip = rf;
        if (lf) 
            leftFootTip = lf;
        groundY = Mathf.Min(rightFootTip ? rightFootTip.position.y : initY - 0.75f, leftFootTip ? leftFootTip.position.y : initY - 0.75f);
        localPhase = globalPhase;

        if (!rightToeEdge) 
            rightToeEdge = FindChildRecursive(transform, "right_toe_edge");
    }

    void FinishEpisode()
    {
        if (!train && localPhase == 3 && debugRunActive)
        {
            debugRunEpisodes++;
            if (episodeScored) debugRunGoals++;
            if (episodeFell) debugRunFalls++;
            if (everKicked) debugRunKicks++;
            if (episodeFirstContactLogged) debugRunContacts++;
            if (debugRunEpisodes % 10 == 0) Debug.Log($"[RUN PROGRESS id={debugRunId}] episodes={debugRunEpisodes} | goals={debugRunGoals} | goalRate={100f * debugRunGoals / debugRunEpisodes:F1}% | falls={debugRunFalls} | isKick={debugRunKicks} | contacts={debugRunContacts}");
        }

        if (!train && debugPerEnvironment && localPhase == 3)
        {
            envDebugEpisodes++;
            if (episodeScored) envDebugGoals++;
            if (episodeFell) envDebugFalls++;
            if (envEpisodeTravel < 0.20f) envDebugNoMove++;
            envDebugLifeSum += epStep * Time.fixedDeltaTime;
            envDebugTravelSum += envEpisodeTravel;

            if (envDebugEpisodes >= envDebugWindow)
            {
                envDebugWindowIndex++;
                float episodeDen = Mathf.Max(1, envDebugEpisodes);
                double speedDen = Mathf.Max(1, envDebugSpeedSamples);
                double actionDen = Mathf.Max(1, envDebugActionValues);
                string modelName = policyModel != null ? policyModel.name : "NULL";
                Debug.Log($"[ENV run={debugRunId} id={diagnosticEnvId} window={envDebugWindowIndex} eps={envDebugWindow}] model={modelName} | goal={100f * envDebugGoals / episodeDen:F0}% | fall={100f * envDebugFalls / episodeDen:F0}% | noMove={100f * envDebugNoMove / episodeDen:F0}% | life={envDebugLifeSum / episodeDen:F1}s | travel={envDebugTravelSum / episodeDen:F2}m | speed={envDebugSpeedSum / speedDen:F2} | decisions={envDebugDecisionCalls / episodeDen:F1}/ep | action={envDebugActionSum / actionDen:F3}");
                envDebugEpisodes = 0;
                envDebugGoals = 0;
                envDebugFalls = 0;
                envDebugNoMove = 0;
                envDebugDecisionCalls = 0;
                envDebugActionValues = 0;
                envDebugSpeedSamples = 0;
                envDebugLifeSum = 0.0;
                envDebugTravelSum = 0.0;
                envDebugSpeedSum = 0.0;
                envDebugActionSum = 0.0;
            }
        }

        if (localPhase == 3)
        {
            p3WindowEpisodes++;
            if (episodeScored)
            {
                p3WindowGoals++;
                p3GoalTimeSum += epStep * Time.fixedDeltaTime;
                p3GoalTimeCount++;
            }
            if (episodeFirstShotEvaluated) p3WindowShotEpisodes++;
            if (episodeFirstShotAccurate) p3WindowAccurateFirstShots++;
            if (everKicked) p3WindowKickEpisodes++;
            if (episodeFell) p3WindowFalls++;
            if (episodeAvoidEntered) p3AvoidEnterEpisodes++;
            if (episodeAvoidExited) p3AvoidExitEpisodes++;
            if (episodeAlignEntered) p3AlignEnterEpisodes++;
            if (episodeAlignExited) p3AlignExitEpisodes++;
            if (episodeAlignReentered) p3AlignReenterEpisodes++;
            if (episodeFirstContactLogged)
            {
                p3FirstContactEpisodes++;
                if (episodeFirstContactState == 4) p3FirstContactFollow++;
                else if (episodeFirstContactState == 3) p3FirstContactKick++;
                else if (episodeFirstContactState == 2) p3FirstContactStrike++;
                else if (episodeFirstContactState == 1) p3FirstContactAlign++;
                else p3FirstContactApproach++;
                if (episodeFirstContactFoot == 2) p3FirstContactBoth++;
                else if (episodeFirstContactFoot == 0) p3FirstContactRight++;
                else p3FirstContactLeft++;
                if (episodeFirstContactBodyAligned) p3FirstContactBodyAligned++;
                if (episodeFirstContactToeAligned) p3FirstContactToeAligned++;
                if (episodeFirstContactRightDominant) p3FirstContactRightDominant++;
                if (episodeFirstContactBallMoving) p3FirstContactBallMoving++;
            }
            if (episodeFirstShotAccurate && episodeFirstContactLogged)
            {
                if (episodeFirstContactState == 4) p3AccurateFollow++;
                else if (episodeFirstContactState == 3) p3AccurateKick++;
                else if (episodeFirstContactState == 2) p3AccurateStrike++;
                else if (episodeFirstContactState == 1) p3AccurateAlign++;
                else p3AccurateApproach++;
                if (episodeFirstContactFoot == 2) p3AccurateBoth++;
                else if (episodeFirstContactFoot == 0) p3AccurateRight++;
                else p3AccurateLeft++;
                if (episodeFirstContactRightDominant) p3AccurateRightDominant++;
            }

            if (p3WindowEpisodes >= 100)
            {
                p3DebugWindowIndex++;
                float goalRate = 100f * p3WindowGoals / p3WindowEpisodes;
                float firstShotPrecision = 100f * p3WindowAccurateFirstShots / Mathf.Max(1, p3WindowShotEpisodes);
                float kickRate = 100f * p3WindowKickEpisodes / p3WindowEpisodes;
                float fallRate = 100f * p3WindowFalls / p3WindowEpisodes;

                Debug.Log($"[P3 run={debugRunId} window={p3DebugWindowIndex} eps=100] goal={goalRate:F1}% | firstShot={firstShotPrecision:F1}% ({p3WindowAccurateFirstShots}/{p3WindowShotEpisodes}) | isKick={kickRate:F1}% | fall={fallRate:F1}%");
                int noContact = p3WindowEpisodes - p3FirstContactEpisodes;
                Debug.Log($"[P3 contact run={debugRunId} window={p3DebugWindowIndex}] state=A/L/S/K/F/N={p3FirstContactApproach}/{p3FirstContactAlign}/{p3FirstContactStrike}/{p3FirstContactKick}/{p3FirstContactFollow}/{noContact} | moving={p3FirstContactBallMoving} | foot=R/L/B={p3FirstContactRight}/{p3FirstContactLeft}/{p3FirstContactBoth} | aligned=body/toe/dominant={p3FirstContactBodyAligned}/{p3FirstContactToeAligned}/{p3FirstContactRightDominant} of {p3FirstContactEpisodes}");
                float accurateApproachRate = 100f * p3AccurateApproach / Mathf.Max(1, p3FirstContactApproach);
                float accurateAlignRate = 100f * p3AccurateAlign / Mathf.Max(1, p3FirstContactAlign);
                float accurateStrikeRate = 100f * p3AccurateStrike / Mathf.Max(1, p3FirstContactStrike);
                float accurateKickRate = 100f * p3AccurateKick / Mathf.Max(1, p3FirstContactKick);
                float accurateFollowRate = 100f * p3AccurateFollow / Mathf.Max(1, p3FirstContactFollow);
                float accurateRightRate = 100f * p3AccurateRight / Mathf.Max(1, p3FirstContactRight);
                float accurateLeftRate = 100f * p3AccurateLeft / Mathf.Max(1, p3FirstContactLeft);
                float accurateBothRate = 100f * p3AccurateBoth / Mathf.Max(1, p3FirstContactBoth);
                float accurateDominantRate = 100f * p3AccurateRightDominant / Mathf.Max(1, p3FirstContactRightDominant);
                Debug.Log($"[P3 quality run={debugRunId} window={p3DebugWindowIndex}] accurateState=A/L/S/K/F={accurateApproachRate:F0}/{accurateAlignRate:F0}/{accurateStrikeRate:F0}/{accurateKickRate:F0}/{accurateFollowRate:F0}% | accurateFoot=R/L/B={accurateRightRate:F0}/{accurateLeftRate:F0}/{accurateBothRate:F0}% | dominant={accurateDominantRate:F0}% ({p3AccurateRightDominant}/{p3FirstContactRightDominant}) | kick=start/stable/contact/fall={p3KickStarts}/{p3KickStableCompletions}/{p3KickContacts}/{p3FallKick}");
                Debug.Log($"[P3 nav run={debugRunId} window={p3DebugWindowIndex}] avoid=enter/exit={p3AvoidEnterEpisodes}/{p3AvoidExitEpisodes} | align=enter/exit/reenter={p3AlignEnterEpisodes}/{p3AlignExitEpisodes}/{p3AlignReenterEpisodes} | fall=A/L/S/K/F={p3FallApproach}/{p3FallAlign}/{p3FallStrike}/{p3FallKick}/{p3FallFollow}");
                double meanGoalTime = p3GoalTimeSum / Mathf.Max(1, p3GoalTimeCount);
                double meanFallSpeed = p3FallSpeedSum / Mathf.Max(1, p3FallDynamicsSamples);
                double meanFallTilt = p3FallTiltSum / Mathf.Max(1, p3FallDynamicsSamples);
                double meanResidual = p3ResidualSum / System.Math.Max(1L, p3ResidualSamples);
                double meanFallResidual = p3FallResidualSum / Mathf.Max(1, p3FallDynamicsSamples);
                Debug.Log($"[P3 dynamics run={debugRunId} window={p3DebugWindowIndex}] goalTime={meanGoalTime:F1}s ({p3GoalTimeCount}) | fallSpeed={meanFallSpeed:F2}m/s | fallTilt={meanFallTilt:F1}deg | residual=avg/fall={meanResidual:F2}/{meanFallResidual:F2}deg");
                ResetP3Metrics(false);
            }
        }

        if (localPhase != globalPhase)
        {
            rwCount = 0;
            rwIdx = 0;
            localPhase = globalPhase;
        }

        float r = GetCumulativeReward();
        recentRewards[rwIdx] = r;
        rwIdx = (rwIdx + 1) % phaseWindow;
        if (rwCount < phaseWindow)
            rwCount++;

        if (rwCount >= phaseWindow)
        {
            float avg = 0f;
            for (int i = 0; i < phaseWindow; i++)
                avg += recentRewards[i];
            avg /= phaseWindow;

            if (globalPhase == 1 && avg >= phase2Threshold)
            {
                globalPhase = 2; localPhase = 2; rwCount = 0; rwIdx = 0;
                Debug.Log($"[SoccerTrain] -> Phase 2 ({avg:F1})");
            }
            else if (globalPhase == 2 && avg >= phase3Threshold)
            {
                globalPhase = 3; localPhase = 3; rwCount = 0; rwIdx = 0;
                Debug.Log($"[SoccerTrain] -> Phase 3 ({avg:F1})");
            }
        }
        EndEpisode();
    }

    public override void OnEpisodeBegin()
    {
        envEpisodeTravel = 0f;
        if (!train) inferenceEpisodeIndex++;
        lastNavProbeState = -1;
        lastNavProbeStep = -1000;
        smoothedStrideScale = 0f;
        tp = 0;
        tq = 0;
        uf1 = 0;
        uf2 = 0;
        uff = 0;
        isKicking = false;
        kickBalanceMaintained = false;
        kickCompletionPending = false;
        stableKickRewarded = false;
        isStrikingPhase = false;
        strikeLockFrames = 0;
        strikeExitFrames = 0;
        strikeRetryUsed = false;
        strikeHeading = Vector3.zero;
        followBallActive = false;
        previousBallVelocity = Vector3.zero;
        shotEvaluationDelay = 0;
        shotContactCooldown = 0;
        pendingShotWasKick = false;
        epStep = 0;
        hasTouchedBall = false;
        maxSwingFootLift = 0;
        everKicked = false;
        episodeScored = false;
        episodeFirstShotEvaluated = false;
        episodeFirstShotAccurate = false;
        episodeFell = false;
        episodeFirstContactLogged = false;
        episodeFirstContactState = -1;
        episodeFirstContactFoot = -1;
        episodeFirstContactBodyAligned = false;
        episodeFirstContactToeAligned = false;
        episodeFirstContactRightDominant = false;
        episodeFirstContactBallMoving = false;
        episodeKickContact = false;
        episodeAvoidEntered = false;
        episodeAvoidExited = false;
        episodeAlignEntered = false;
        episodeAlignExited = false;
        episodeAlignReentered = false;
        episodeAlignEntries = 0;
        previousAvoiding = false;
        previousShootSpotLocked = false;
        prevDistToShootSpot = -1f;
        reachedShootSpotOnce = false;
        shootSpotLocked = false;
        shootSpotTimer = 0;
        shootSpotStableFrames = 0;
        alignExitFrames = 0;
        approachTargetLocked = false;
        lockedApproachTarget = Vector3.zero;
        lockedApproachBallPos = Vector3.zero;
        lockedApproachDir = Vector3.zero;
        previousRightToePosition = Vector3.zero;
        rightToeVelocity = Vector3.zero;
        rightToeVelocityReady = false;

        for (int i = 0; i < numJoints; i++)
        {
            gaitU[i] = 0f;
            gaitTotal[i] = 0f;
        }
        switch (localPhase)
        {
            case 1:
                Vector3 p1Offset = new Vector3(Random.Range(-0.5f, 0.5f), 0, Random.Range(-0.5f, 0.5f));
                if (transform.parent != null)
                    p1Offset = transform.parent.TransformDirection(p1Offset);
                Teleport(spawnPos + p1Offset, Random.Range(-30f, 30f));
                SpawnBall(new Vector3(0f, -10f, 0f));
                break;
            case 2:
                Teleport(spawnPos, Random.Range(-15f, 15f));
                SpawnBallNear();
                break;
            case 3:
                float safeW = fieldHalfW - 1.0f;
                float safeL = fieldLength - 2.0f;
                float randForward = Random.Range(1.0f, safeL);
                float randLateral = Random.Range(-safeW, safeW);
                Vector3 safePos = spawnPos + goalDir * randForward + fieldRight * randLateral;
                Teleport(safePos, Random.Range(-30f, 30f));
                SpawnBallFar();
                break;
        }
    }

    void Teleport(Vector3 p, float y)
    {
        p.y = initY;

        if (rootBody != null)
        {
            rootBody.TeleportRoot(p, Quaternion.LookRotation(goalDir) * Quaternion.Euler(0f, y, 0f));
            rootBody.velocity = Vector3.zero;
            rootBody.angularVelocity = Vector3.zero;
        }

        if (allBodies != null)
        {
            foreach (var ab in allBodies)
            {
                if (ab.isRoot) continue;
                ab.velocity = Vector3.zero;
                ab.angularVelocity = Vector3.zero;

                if (ab.jointPosition.dofCount > 0)
                {
                    var jp = ab.jointPosition;
                    float targetDeg = 0f;
                    string jn = ab.name;

                    if (jn == "right_hip_pitch_link" || jn == "left_hip_pitch_link") 
                        targetDeg = -6f;
                    else if (jn == "right_knee_link" || jn == "left_knee_link") 
                        targetDeg = 12f;
                    else if (jn == "right_ankle_pitch_link" || jn == "left_ankle_pitch_link") 
                        targetDeg = -6f;
                    else if (jn == "right_hip_roll_link") 
                        targetDeg = -3f;
                    else if (jn == "left_hip_roll_link") 
                        targetDeg = 3f;
                    else if (jn == "right_ankle_roll_link") 
                        targetDeg = 3f;
                    else if (jn == "left_ankle_roll_link") 
                        targetDeg = -3f;

                    else if (jn == "right_shoulder_pitch_link" || jn == "left_shoulder_pitch_link") 
                        targetDeg = 10f;
                    else if (jn == "right_elbow_link" || jn == "left_elbow_link") 
                        targetDeg = 70f;
                    else if (jn == "right_shoulder_roll_link") 
                        targetDeg = -5f;
                    else if (jn == "left_shoulder_roll_link") 
                        targetDeg = 5f;

                    for (int i = 0; i < jp.dofCount; i++)
                        jp[i] = targetDeg * Mathf.Deg2Rad;
                    ab.jointPosition = jp;

                    var jv = ab.jointVelocity;
                    for (int i = 0; i < jv.dofCount; i++)
                        jv[i] = 0f;
                    ab.jointVelocity = jv;
                }
            }
        }
    }

    void SpawnBallNear()
    {
        SpawnBall(new Vector3(Random.Range(-3f, -0.5f), BALL_R, Random.Range(-3f, 3f)));
    }

    void SpawnBallFar()
    {
        SpawnBall(new Vector3(Random.Range(-5f, 5f), BALL_R, Random.Range(-4f, 4f)));
    }

    void SpawnBall(Vector3 p)
    {
        if (ball)
        {
            ball.localPosition = p;
            if (ballRb)
            {
                ballRb.velocity = Vector3.zero;
                ballRb.angularVelocity = Vector3.zero;
                ballRb.isKinematic = (p.y < -5f);
            }
        }
    }

    public override void CollectObservations(VectorSensor s)
    {
        if (rootBody == null || revoluteJoints == null)
            return;
        var t = rootBody.transform;

        s.AddObservation(t.InverseTransformDirection(Vector3.down));
        s.AddObservation(t.InverseTransformDirection(rootBody.angularVelocity));
        s.AddObservation(t.InverseTransformDirection(rootBody.velocity));

        for (int i = 0; i < 12; i++)
        {
            if (legIdx[i] >= 0)
            {
                var joint = revoluteJoints[legIdx[i]];
                s.AddObservation(joint.jointPosition.dofCount > 0 ? joint.jointPosition[0] : 0f);
                s.AddObservation(joint.jointVelocity.dofCount > 0 ? joint.jointVelocity[0] : 0f);
            }
            else
            {
                s.AddObservation(0f);
                s.AddObservation(0f);
            }
        }

        s.AddObservation(t.position.y - groundY);

        if (ball && localPhase > 1)
        {
            s.AddObservation(t.InverseTransformPoint(ball.position));
            s.AddObservation(t.InverseTransformDirection(ballRb ? ballRb.velocity : Vector3.zero));
        }
        else
        {
            s.AddObservation(Vector3.zero);
            s.AddObservation(Vector3.zero);
        }

        s.AddObservation(t.InverseTransformDirection(targetMoveDir));
        s.AddObservation(t.InverseTransformDirection(targetLookDir));

        s.AddObservation(uf1);
        s.AddObservation(uf2);

        for (int i = 0; i < 12; i++)
            s.AddObservation(legIdx[i] >= 0 ? gaitU[legIdx[i]] : 0f);

        s.AddObservation(Mathf.Clamp(targetVel, 0f, 0.8f) - 0.8f);
    }

    public override void OnActionReceived(ActionBuffers a)
    {
        if (!train && policyModel != null)
            SetModel("gewu", policyModel);
        if (rootBody == null || revoluteJoints == null || revoluteJoints.Length == 0)
            return;

        var ca = a.ContinuousActions;
        if (!train && debugPerEnvironment && localPhase == 3)
        {
            envDebugDecisionCalls++;
            for (int i = 0; i < ca.Length; i++)
            {
                envDebugActionSum += Mathf.Abs(ca[i]);
                envDebugActionValues++;
            }
        }
        float kk = 0.9f;

        int nLegActions = Mathf.Min(12, ca.Length);
        for (int i = 0; i < nLegActions; i++)
            if (legIdx[i] >= 0)
                gaitU[legIdx[i]] = gaitU[legIdx[i]] * kk + (1f - kk) * ca[i];
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        for (int i = 0; i < ca.Length; i++)
            ca[i] = 0f;
    }

    void SetDrive(ArticulationBody j, float d, float s, float mp, float f)
    {
        if (!j)
            return;
        var dr = j.xDrive;
        dr.stiffness = s;
        dr.damping = mp;
        dr.forceLimit = f;
        dr.target = d;
        j.xDrive = dr;
    }

    bool TryGetRightToeEdge(Vector3 queryPoint, out Vector3 closestPoint, out Vector3 forwardDir)
    {
        closestPoint = Vector3.zero;
        forwardDir = Vector3.zero;

        if (!rightToeEdge) 
            return false;

        Vector3 edgeDir = rightToeEdge.right;
        edgeDir.y = 0f;

        if (edgeDir.sqrMagnitude < 0.0001f) 
            return false;

        edgeDir.Normalize();

        Vector3 offset = queryPoint - rightToeEdge.position;
        offset.y = 0f;
        float alongEdge = Mathf.Clamp(Vector3.Dot(offset, edgeDir), -rightToeHalfWidth, rightToeHalfWidth);
        closestPoint = rightToeEdge.position + edgeDir * alongEdge;
        forwardDir = rightToeEdge.forward;
        forwardDir.y = 0f;

        if (forwardDir.sqrMagnitude < 0.0001f) 
            return false;

        forwardDir.Normalize();
        return true;
    }

    void FixedUpdate()
    {
        if (train) trainStepCounter += 1.0;

        var tRoot = rootBody.transform;
        Vector3 pos = tRoot.position;
        if (!train && debugPerEnvironment && localPhase == 3)
        {
            Vector3 velocity = rootBody.velocity;
            velocity.y = 0f;
            envEpisodeTravel += velocity.magnitude * Time.fixedDeltaTime;
            envDebugSpeedSum += velocity.magnitude;
            envDebugSpeedSamples++;
        }
        float bodyPitch = Mathf.Asin(Mathf.Clamp(tRoot.forward.y, -1f, 1f)) * Mathf.Rad2Deg;
        if (rightToeEdge)
        {
            if (rightToeVelocityReady) rightToeVelocity = (rightToeEdge.position - previousRightToePosition) / Mathf.Max(Time.fixedDeltaTime, 0.0001f);
            previousRightToePosition = rightToeEdge.position;
            rightToeVelocityReady = true;
        }
        epStep++;
        contactStateBeforeNavigation = isKicking ? 3 : (isStrikingPhase ? 2 : (shootSpotLocked ? 1 : (followBallActive ? 4 : 0)));

        float warmUp = Mathf.Clamp01(epStep / 30f);

        targetMoveDir = Vector3.zero;
        targetLookDir = Vector3.zero;
        float distToBallFlat = 10f;

        float spotBrake = 1.0f;
        bool closeDynamicTurnRisk = false;

        if (ball != null)
        {
            Vector2 flatPos = new Vector2(pos.x, pos.z);
            Vector2 flatBall = new Vector2(ball.position.x, ball.position.z);
            distToBallFlat = Vector2.Distance(flatPos, flatBall);
        }

        if (localPhase >= 2 && ball != null)
        {
            Vector3 toBallDir = (ball.position - pos).normalized;
            toBallDir.y = 0f;
            toBallDir.Normalize();

            if (localPhase == 3)
            {
                Vector3 effectiveGoalPos = goalLeftPos;
                float bFwd = Vector3.Dot(ball.position - spawnPos, goalDir);
                float bLat = Vector3.Dot(ball.position - spawnPos, fieldRight);

                if (bFwd > fieldLength - 1.5f && Mathf.Abs(bLat) > 1.65f) 
                    effectiveGoalPos -= goalDir * 2.5f;

                Vector3 ballVelFlat = ballRb ? new Vector3(ballRb.velocity.x, 0f, ballRb.velocity.z) : Vector3.zero;
                Vector3 rootVelFlat = new Vector3(rootBody.velocity.x, 0f, rootBody.velocity.z);
                float ballSpeed = ballVelFlat.magnitude;
                float predictionTime = Mathf.Clamp(0.12f + distToBallFlat * 0.12f, 0.12f, 0.40f);
                Vector3 predictedBallPos = ball.position;

                if (ballSpeed > 0.08f) 
                    predictedBallPos += Vector3.ClampMagnitude(ballVelFlat * predictionTime, 0.6f);

                float predLat = Vector3.Dot(predictedBallPos - spawnPos, fieldRight);
                float maxSafeLat = fieldHalfW - 0.6f;

                if (Mathf.Abs(predLat) > maxSafeLat) 
                    predictedBallPos -= fieldRight * (predLat - Mathf.Sign(predLat) * maxSafeLat);

                Vector3 ballToGoalDir = effectiveGoalPos - predictedBallPos;
                ballToGoalDir.y = 0f;

                if (ballToGoalDir.sqrMagnitude < 0.0001f) 
                    ballToGoalDir = goalDir;

                ballToGoalDir.Normalize();
                Vector3 attackHeading = isStrikingPhase && strikeHeading.sqrMagnitude > 0.0001f ? strikeHeading : ballToGoalDir;

                Vector3 rightDir = Vector3.Cross(Vector3.up, ballToGoalDir).normalized;
                Vector3 rootForwardFlat = tRoot.forward;
                rootForwardFlat.y = 0f;

                if (rootForwardFlat.sqrMagnitude < 0.0001f) 
                    rootForwardFlat = ballToGoalDir;

                rootForwardFlat.Normalize();

                float localZ = Vector3.Dot(pos - predictedBallPos, ballToGoalDir);
                float localX = Vector3.Dot(pos - predictedBallPos, rightDir);
                float lateralError = Mathf.Abs(localX + 0.08f);
                float actualLocalZ = Vector3.Dot(pos - ball.position, ballToGoalDir);
                float actualLocalX = Vector3.Dot(pos - ball.position, rightDir);
                float actualLateralError = Mathf.Abs(actualLocalX + 0.08f);
                float angleToGoal = Vector3.Angle(rootForwardFlat, ballToGoalDir);
                float angleToAttackHeading = Vector3.Angle(rootForwardFlat, attackHeading);
                Vector3 attackRightDir = Vector3.Cross(Vector3.up, attackHeading).normalized;
                float strikeLocalZ = Vector3.Dot(pos - ball.position, attackHeading);
                float strikeLateralError = Mathf.Abs(Vector3.Dot(pos - ball.position, attackRightDir) + 0.08f);
                float distToPredictedBall = Vector2.Distance(new Vector2(pos.x, pos.z), new Vector2(predictedBallPos.x, predictedBallPos.z));
                float ballAngleToGoal = ballSpeed > 0.1f ? Vector3.Angle(ballVelFlat, ballToGoalDir) : 180f;
                float distToGoal = Vector3.Distance(predictedBallPos, effectiveGoalPos);
                float maxAllowedAngle = Mathf.Clamp(Mathf.Atan2(1.1f, Mathf.Max(0.1f, distToGoal)) * Mathf.Rad2Deg, 3f, 30f);
                bool isBallOnTrack = ballSpeed > 0.25f && ballAngleToGoal < maxAllowedAngle;

                Vector3 idealShootSpot = predictedBallPos - ballToGoalDir * 0.6f - rightDir * 0.08f;
                idealShootSpot.y = pos.y;

                Vector3 candidateToShootSpot = idealShootSpot - pos;
                candidateToShootSpot.y = 0f;
                float candidateSpotDistance = candidateToShootSpot.magnitude;
                Vector3 approachBallShift = ball.position - lockedApproachBallPos;
                approachBallShift.y = 0f;
                bool approachLockInvalid = approachTargetLocked && (approachBallShift.magnitude > 0.16f || ballSpeed > 0.30f);
                if (approachLockInvalid)
                {
                    approachTargetLocked = false;
                    lockedApproachDir = Vector3.zero;
                }
                if (!approachTargetLocked && !shootSpotLocked && !isStrikingPhase && !followBallActive && ballSpeed < 0.18f && candidateSpotDistance < 1.0f && localZ < -0.30f && lateralError < 0.35f)
                {
                    approachTargetLocked = true;
                    lockedApproachTarget = idealShootSpot;
                    lockedApproachBallPos = ball.position;
                    lockedApproachDir = candidateToShootSpot.sqrMagnitude > 0.0001f ? candidateToShootSpot.normalized : ballToGoalDir;
                }
                if (approachTargetLocked) idealShootSpot = lockedApproachTarget;

                Vector3 toShootSpot = idealShootSpot - pos;
                toShootSpot.y = 0f;
                float distToShootSpotFlat = toShootSpot.magnitude;
                Vector3 approachDir = approachTargetLocked && lockedApproachDir.sqrMagnitude > 0.0001f ? lockedApproachDir : (toShootSpot.sqrMagnitude > 0.0001f ? toShootSpot.normalized : ballToGoalDir);
                float signedSpotRemaining = Vector3.Dot(idealShootSpot - pos, approachDir);
                float approachSpeed = Mathf.Max(0f, Vector3.Dot(rootVelFlat, approachDir));
                float captureSpeed = Mathf.Max(approachSpeed, rootVelFlat.magnitude * 0.5f);
                float captureRadius = Mathf.Clamp(0.15f + captureSpeed * 0.28f, 0.15f, 0.30f);
                bool captureLaneReady = actualLocalZ < -0.36f && actualLocalZ > -0.96f && actualLateralError < 0.16f;
                bool alignEnvelopeReady = actualLocalZ < -0.26f && actualLocalZ > -1.12f && actualLateralError < 0.30f;
                bool alignPoseReady = actualLocalZ < -0.34f && actualLocalZ > -0.88f && actualLateralError < 0.16f;
                bool stateChangedThisFrame = false;

                if (!shootSpotLocked && shootSpotTimer < 0) shootSpotTimer++;
                if (followBallActive && (ballSpeed < 0.10f || ballAngleToGoal > Mathf.Max(12f, maxAllowedAngle * 1.5f)))
                {
                    followBallActive = false;
                    stateChangedThisFrame = true;
                }

                bool isAvoiding = false;
                if (isStrikingPhase && !isKicking)
                {
                    strikeLockFrames--;
                    bool strikeTooFar = distToBallFlat > 1.15f;
                    bool strikePassedBall = strikeLocalZ > 0.14f;
                    bool strikeLateralLost = strikeLateralError > 0.22f;
                    bool strikeYawLost = angleToAttackHeading > 15f;
                    bool strikeTimedOut = strikeLockFrames <= 0;
                    bool strikeSoftViolation = strikeTooFar || strikePassedBall || strikeLateralLost || strikeYawLost;
                    strikeExitFrames = strikeSoftViolation ? strikeExitFrames + 1 : 0;
                    bool lostStrikeControl = strikeExitFrames >= 12;

                    bool strikeStillUseful = !strikeSoftViolation && strikeLocalZ < -0.10f && strikeLocalZ > -0.80f && strikeLateralError < 0.16f && angleToAttackHeading < 12f;
                    if (strikeTimedOut && strikeStillUseful && !strikeRetryUsed)
                    {
                        strikeLockFrames = 100;
                        strikeRetryUsed = true;
                        strikeTimedOut = false;
                    }

                    if (isBallOnTrack || lostStrikeControl || strikeTimedOut)
                    {
                        if (debugNavigation && !train && navProbeAgent == this)
                        {
                            string exitReason = $"{(isBallOnTrack ? "track+" : "")}{(strikeTooFar ? "distance+" : "")}{(strikePassedBall ? "passed+" : "")}{(strikeLateralLost ? "lateral+" : "")}{(strikeYawLost ? "yaw+" : "")}{(strikeTimedOut ? "timeout+" : "")}".TrimEnd('+');
                            Debug.Log($"[NAVEXIT run={debugRunId} env={diagnosticEnvId} ep={inferenceEpisodeIndex} from=Strike reason={exitReason}] ballSpeed={ballSpeed:F2} ballAngle={ballAngleToGoal:F0}/{maxAllowedAngle:F0} predDist={distToPredictedBall:F2} predZ={localZ:F2} predXErr={lateralError:F2} actualZ={actualLocalZ:F2} actualXErr={actualLateralError:F2} yaw={angleToAttackHeading:F0} bad={strikeExitFrames} lock={strikeLockFrames}");
                        }

                        isStrikingPhase = false;
                        strikeExitFrames = 0;
                        strikeRetryUsed = false;
                        strikeHeading = Vector3.zero;
                        approachTargetLocked = false;
                        lockedApproachDir = Vector3.zero;
                        followBallActive = isBallOnTrack;
                        if (!isBallOnTrack) shootSpotTimer = -25;
                        stateChangedThisFrame = true;
                    }
                }

                if (shootSpotLocked)
                {
                    shootSpotTimer++;
                    bool alignPositionLost = !alignEnvelopeReady;
                    bool alignBallLost = ballSpeed > 0.35f && !isBallOnTrack;
                    bool alignSoftViolation = alignPositionLost || alignBallLost;
                    alignExitFrames = alignSoftViolation ? alignExitFrames + 1 : 0;

                    if (isBallOnTrack)
                    {
                        if (debugNavigation && !train && navProbeAgent == this) Debug.Log($"[NAVEXIT run={debugRunId} env={diagnosticEnvId} ep={inferenceEpisodeIndex} from=Align reason=track] ballSpeed={ballSpeed:F2} ballAngle={ballAngleToGoal:F0}/{maxAllowedAngle:F0} actualZ={actualLocalZ:F2} actualXErr={actualLateralError:F2} yaw={angleToGoal:F0} speed={rootVelFlat.magnitude:F2} timer={shootSpotTimer} stable={shootSpotStableFrames}");
                        shootSpotLocked = false;
                        approachTargetLocked = false;
                        lockedApproachDir = Vector3.zero;
                        followBallActive = true;
                        shootSpotTimer = 0;
                        shootSpotStableFrames = 0;
                        alignExitFrames = 0;
                        stateChangedThisFrame = true;
                    }
                    else if (alignExitFrames >= 12)
                    {
                        if (debugNavigation && !train && navProbeAgent == this)
                        {
                            string exitReason = alignPositionLost ? "envelope" : "ballSpeed";
                            Debug.Log($"[NAVEXIT run={debugRunId} env={diagnosticEnvId} ep={inferenceEpisodeIndex} from=Align reason={exitReason}] ballSpeed={ballSpeed:F2} actualZ={actualLocalZ:F2} actualXErr={actualLateralError:F2} predZ={localZ:F2} predXErr={lateralError:F2} yaw={angleToGoal:F0} speed={rootVelFlat.magnitude:F2} bad={alignExitFrames} timer={shootSpotTimer} stable={shootSpotStableFrames}");
                        }

                        shootSpotLocked = false;
                        approachTargetLocked = false;
                        lockedApproachDir = Vector3.zero;
                        shootSpotTimer = -35;
                        shootSpotStableFrames = 0;
                        alignExitFrames = 0;
                        stateChangedThisFrame = true;
                    }
                    else if (shootSpotTimer > 240)
                    {
                        if (debugNavigation && !train && navProbeAgent == this) Debug.Log($"[NAVEXIT run={debugRunId} env={diagnosticEnvId} ep={inferenceEpisodeIndex} from=Align reason=timeout] ballSpeed={ballSpeed:F2} actualZ={actualLocalZ:F2} actualXErr={actualLateralError:F2} predZ={localZ:F2} predXErr={lateralError:F2} yaw={angleToGoal:F0} speed={rootVelFlat.magnitude:F2} timer={shootSpotTimer} stable={shootSpotStableFrames}");
                        shootSpotLocked = false;
                        approachTargetLocked = false;
                        lockedApproachDir = Vector3.zero;
                        shootSpotTimer = -40;
                        shootSpotStableFrames = 0;
                        alignExitFrames = 0;
                        stateChangedThisFrame = true;
                    }
                }

                bool passedShootSpot = approachTargetLocked && signedSpotRemaining <= 0.03f && alignEnvelopeReady;
                bool reachedShootSpot = ballSpeed < 0.25f && captureLaneReady && (distToShootSpotFlat < captureRadius || passedShootSpot);

                if (!stateChangedThisFrame && !followBallActive && !isStrikingPhase && !shootSpotLocked && shootSpotTimer >= 0 && reachedShootSpot)
                {
                    shootSpotLocked = true;
                    approachTargetLocked = false;
                    lockedApproachDir = Vector3.zero;
                    shootSpotTimer = 0;
                    shootSpotStableFrames = 0;
                    alignExitFrames = 0;
                    stateChangedThisFrame = true;

                    if (!reachedShootSpotOnce)
                    {
                        reachedShootSpotOnce = true;
                        AddReward(shootSpotArrivalBonus);
                    }
                }

                if (shootSpotLocked)
                {
                    bool poseStable = alignPoseReady && angleToGoal < 10f && rootVelFlat.magnitude < 0.28f;
                    shootSpotStableFrames = poseStable ? shootSpotStableFrames + 1 : Mathf.Max(0, shootSpotStableFrames - 1);

                    if (!stateChangedThisFrame && shootSpotStableFrames >= 8)
                    {
                        shootSpotLocked = false;
                        shootSpotTimer = 0;
                        shootSpotStableFrames = 0;
                        isStrikingPhase = true;
                        strikeLockFrames = 240;
                        strikeExitFrames = 0;
                        strikeRetryUsed = false;
                        strikeHeading = ballToGoalDir;
                        stateChangedThisFrame = true;
                    }
                }

                if (isStrikingPhase)
                {
                    targetMoveDir = attackHeading;
                    targetLookDir = attackHeading;
                    spotBrake = distToBallFlat < 0.38f ? 0.35f : 0.65f;
                }
                else if (shootSpotLocked)
                {
                    targetMoveDir = Vector3.zero;
                    targetLookDir = ballToGoalDir;
                    spotBrake = 0f;
                }
                else if (followBallActive)
                {
                    float followGap = -actualLocalZ;
                    bool followCanAdvance = followGap > 0.48f;
                    targetMoveDir = followCanAdvance ? ballToGoalDir : Vector3.zero;
                    targetLookDir = followCanAdvance ? ballToGoalDir : rootForwardFlat;
                    spotBrake = followCanAdvance ? Mathf.Lerp(0.15f, 0.65f, Mathf.InverseLerp(0.48f, 1.0f, followGap)) : 0f;
                }
                else
                {
                    bool closeRecoveryHold = shootSpotTimer < 0 && distToPredictedBall < 0.85f;
                    isAvoiding = !closeRecoveryHold && ballSpeed < 0.12f && localZ > -0.3f && distToPredictedBall < 1.5f;

                    if (closeRecoveryHold)
                    {
                        targetMoveDir = Vector3.zero;
                        targetLookDir = rootForwardFlat;
                        spotBrake = 0f;
                    }
                    else if (isAvoiding)
                    {
                        float sideSign = localX >= 0f ? 1f : -1f;
                        float ballLat = Vector3.Dot(predictedBallPos - spawnPos, fieldRight);

                        if (ballLat > fieldHalfW - 1.2f) 
                            sideSign = -1f;
                        else if (ballLat < -fieldHalfW + 1.2f) 
                            sideSign = 1f;

                        Vector3 navTarget = predictedBallPos + rightDir * sideSign * 0.8f - ballToGoalDir * 0.6f;
                        navTarget.y = pos.y;
                        targetMoveDir = (navTarget - pos).normalized;
                        targetLookDir = targetMoveDir;
                    }
                    else
                    {
                        targetMoveDir = toShootSpot.sqrMagnitude > 0.0001f ? toShootSpot.normalized : Vector3.zero;
                        float moveGoalAngle = targetMoveDir.sqrMagnitude > 0.0001f ? Vector3.Angle(targetMoveDir, ballToGoalDir) : 0f;
                        bool finalApproach = localZ < -0.3f && distToPredictedBall < 1.2f && lateralError < 0.30f && moveGoalAngle < 18f;
                        targetLookDir = finalApproach ? ballToGoalDir : targetMoveDir;
                        float headingScale = finalApproach ? Mathf.Lerp(0.50f, 1f, Mathf.InverseLerp(45f, 12f, angleToGoal)) : 1f;

                        if (approachTargetLocked)
                        {
                            float swingPhase = Mathf.Max(uf1, uf2);
                            float phaseCarryDistance = approachSpeed * Time.fixedDeltaTime * Mathf.Lerp(2f, 5f, swingPhase);
                            float speedDistance = Mathf.Max(0f, signedSpotRemaining - 0.05f - phaseCarryDistance);
                            float speedPressure = Mathf.Clamp01((approachSpeed * approachSpeed - 0.04f) / 0.32f);
                            float brakeCurve = Mathf.Lerp(0.50f, 0.32f, speedPressure);
                            float desiredApproachSpeed = Mathf.Clamp(Mathf.Sqrt(brakeCurve * speedDistance), 0.06f, 0.45f);
                            float distanceScale = desiredApproachSpeed / 0.45f;
                            float closingScale = desiredApproachSpeed / Mathf.Max(0.25f, approachSpeed);
                            spotBrake = Mathf.Clamp(Mathf.Min(distanceScale, closingScale) * headingScale, 0.10f, 1f);
                            float preCaptureBrake = Mathf.Lerp(0.45f, 1f, Mathf.InverseLerp(captureRadius, captureRadius + 0.30f, signedSpotRemaining));
                            spotBrake = Mathf.Min(spotBrake, preCaptureBrake);
                        }
                        else
                        {
                            float distanceBrake = Mathf.Lerp(0.35f, 1f, Mathf.InverseLerp(0.35f, 0.75f, distToShootSpotFlat));
                            spotBrake = Mathf.Clamp(distanceBrake * headingScale, 0.10f, 1f);
                        }
                    }
                }

                if (isAvoiding && !previousAvoiding) episodeAvoidEntered = true;
                if (!isAvoiding && previousAvoiding) episodeAvoidExited = true;
                previousAvoiding = isAvoiding;

                if (shootSpotLocked && !previousShootSpotLocked)
                {
                    episodeAlignEntries++;
                    episodeAlignEntered = true;
                    if (episodeAlignEntries > 1) episodeAlignReentered = true;
                }
                if (!shootSpotLocked && previousShootSpotLocked) episodeAlignExited = true;
                previousShootSpotLocked = shootSpotLocked;

                int navStateId = isKicking ? 3 : (isStrikingPhase ? 2 : (shootSpotLocked ? 1 : (followBallActive ? 4 : 0)));
                bool navStateChanged = navStateId != lastNavProbeState;
                bool navPeriodicSample = distToBallFlat < 1.2f && epStep - lastNavProbeStep >= 150;
                if (debugNavigation && !train && navProbeAgent == this && distToBallFlat < 1.6f && (navStateChanged || navPeriodicSample))
                {
                    string navState = navStateId == 4 ? "Follow" : (navStateId == 3 ? "Kick" : (navStateId == 2 ? "Strike" : (navStateId == 1 ? "Align" : "Approach")));
                    float moveYaw = targetMoveDir.sqrMagnitude > 0.0001f ? Vector3.SignedAngle(rootForwardFlat, targetMoveDir, Vector3.up) : 0f;
                    Debug.Log($"[NAV8 run={debugRunId} env={diagnosticEnvId} ep={inferenceEpisodeIndex} state={navState}] avoid={isAvoiding} targetLock={approachTargetLocked} spot={distToShootSpotFlat:F2} remain={signedSpotRemaining:F2} close={approachSpeed:F2} z={actualLocalZ:F2} x={actualLocalX:F2} yaw={angleToGoal:F0} move={moveYaw:F0} speed={rootVelFlat.magnitude:F2} stride={smoothedStrideScale:F2}");
                    lastNavProbeState = navStateId;
                    lastNavProbeStep = epStep;
                }

                if (!isStrikingPhase)
                {
                    if (prevDistToShootSpot >= 0f)
                    {
                        float deltaDist = Mathf.Clamp(prevDistToShootSpot - distToShootSpotFlat, -0.05f, 0.05f);
                        AddReward(deltaDist * shootSpotProgressCoef);
                    }

                    prevDistToShootSpot = distToShootSpotFlat;
                }
                else
                    prevDistToShootSpot = -1f;


                if (isStrikingPhase && !isKicking && kickWaitTimer <= 0 && epStep > 60)
                {
                    float kickTilt = Vector3.Angle(tRoot.up, Vector3.up);
                    Vector3 contactBallPos = ball.position;

                    if (ballSpeed > 0.08f) contactBallPos += Vector3.ClampMagnitude(ballVelFlat * 0.18f, 0.30f);

                    bool toePoseReady = false;
                    Vector3 toeClosestPoint;
                    Vector3 toeForwardDir;

                    if (TryGetRightToeEdge(contactBallPos, out toeClosestPoint, out toeForwardDir))
                    {
                        Vector3 toeCenterToBall = contactBallPos - rightToeEdge.position;
                        toeCenterToBall.y = 0f;
                        Vector3 toeEdgeDir = rightToeEdge.right;
                        toeEdgeDir.y = 0f;

                        if (toeEdgeDir.sqrMagnitude > 0.0001f)
                        {
                            toeEdgeDir.Normalize();
                            float toeForwardDistance = Vector3.Dot(toeCenterToBall, toeForwardDir);
                            float toeCenterError = Mathf.Abs(Vector3.Dot(toeCenterToBall, toeEdgeDir));
                            float toeYawError = Vector3.Angle(toeForwardDir, attackHeading);
                            toePoseReady = toeForwardDistance > BALL_R + 0.02f && toeForwardDistance < 0.32f && toeCenterError <= rightToeHalfWidth && toeYawError < 10f;
                        }
                    }

                    bool kickSetupReady = actualLocalZ < -0.08f && actualLateralError < 0.12f && angleToAttackHeading < 10f && kickTilt < 15f && distToBallFlat < 0.55f;

                    if (kickSetupReady)
                    {
                        float setupBrake = Mathf.Lerp(0.16f, 0.45f, Mathf.InverseLerp(0.28f, 0.55f, distToBallFlat));
                        spotBrake = Mathf.Min(spotBrake, setupBrake);
                    }

                    bool kickGeometryReady = actualLocalZ < -0.10f && actualLateralError < 0.10f && angleToAttackHeading < 8f && kickTilt < 12f && toePoseReady && distToBallFlat < 0.38f;
                    float leftFootHeight = leftFootTip ? leftFootTip.position.y - groundY : 1f;
                    float rightFootHeight = rightFootTip ? rightFootTip.position.y - groundY : 1f;
                    bool physicalDoubleSupport = leftFootHeight < 0.06f && rightFootHeight < 0.06f;
                    bool doubleSupportReady = uf1 < 0.12f && uf2 < 0.12f && physicalDoubleSupport;

                    if (kickGeometryReady) spotBrake = Mathf.Min(spotBrake, 0.10f);

                    bool kickPoseReady = kickGeometryReady && rootVelFlat.magnitude < 0.22f;

                    if (kickPoseReady && doubleSupportReady)
                    {
                        isKicking = true;
                        if (localPhase == 3) p3KickStarts++;
                        kickBalanceMaintained = true;
                        kickCompletionPending = false;
                        tq = 1;
                        tp = 0;
                        uf1 = 0f;
                        uf2 = 0f;

                        if (!everKicked) 
                            AddReward(0.10f);

                        everKicked = true;
                    }
                }

                float closeTargetYaw = Mathf.Abs(Vector3.SignedAngle(rootForwardFlat, targetLookDir, Vector3.up));
                closeDynamicTurnRisk = !isStrikingPhase && distToBallFlat < 0.60f && closeTargetYaw > 10f;
            }
            else
            {
                targetMoveDir = toBallDir;
                targetLookDir = toBallDir;
            }
        }
        else
        {
            float forwardProgress = Vector3.Dot(pos - spawnPos, goalDir);
            Vector3 lookAheadPoint = spawnPos + goalDir * (forwardProgress + 2.0f);
            Vector3 toGoalDir = (goalLeftPos - pos).normalized;
            Vector3 toTargetDir = (lookAheadPoint - pos).normalized;
            toTargetDir.y = 0f;
            toTargetDir.Normalize();
            targetMoveDir = toTargetDir;
            targetLookDir = toTargetDir;
        }

        if (isKicking)
        {
            float kickTiltNow = Vector3.Angle(tRoot.up, Vector3.up);
            float kickPitchNow = Mathf.Abs(Mathf.Asin(Mathf.Clamp(tRoot.forward.y, -1f, 1f)) * Mathf.Rad2Deg);
            float kickRollNow = Mathf.Abs(Mathf.Asin(Mathf.Clamp(tRoot.right.y, -1f, 1f)) * Mathf.Rad2Deg);
            kickBalanceMaintained &= kickTiltNow < 24f && kickPitchNow < 24f && kickRollNow < 22f;
            tq++;
            uff = (-Mathf.Cos(Mathf.PI * 2f * tq / T2) + 1f) / 2f;
            if (tq >= T2)
            {
                isKicking = false;
                if (localPhase == 3 && kickBalanceMaintained) p3KickStableCompletions++;
                kickCompletionPending = kickBalanceMaintained;
                kickBalanceMaintained = false;
                tq = 0;
                kickWaitTimer = 30;
            }
        }
        if (kickWaitTimer > 0)
            kickWaitTimer--;

        if (!isKicking)
        {
            tp++;

            if (tp > 0 && tp <= T1)
            {
                tp0 = tp;
                uf1 = (-Mathf.Cos(Mathf.PI * 2f * tp0 / T1) + 1f) / 2f;
                uf2 = 0f;
            }

            if (tp > T1 && tp <= 2 * T1)
            {
                tp0 = tp - T1;
                uf1 = 0f;
                uf2 = (-Mathf.Cos(Mathf.PI * 2f * tp0 / T1) + 1f) / 2f;
            }

            if (tp >= 2 * T1) 
                tp = 0;
        }

        currentSpeedThreshold = 0.6f;
        if (localPhase >= 2 && ball != null)
        {
            if (isStrikingPhase)
                currentSpeedThreshold = 0.3f;
            else if (distToBallFlat < 1.5f)
                currentSpeedThreshold = Mathf.Lerp(0.3f, 0.6f, (distToBallFlat - 0.3f) / 1.2f);
        }
        currentSpeedThreshold = Mathf.Clamp(currentSpeedThreshold, 0.3f, 0.6f);

        float targetStrideScale = Mathf.Clamp(currentSpeedThreshold / 0.6f, 0.2f, 1.0f);
        targetStrideScale *= spotBrake;
        if (closeDynamicTurnRisk)
        {
            float closeMoveScale = Mathf.Lerp(0.30f, 1f, Mathf.InverseLerp(0.30f, 0.60f, distToBallFlat));
            targetStrideScale *= closeMoveScale;
        }

        if (targetMoveDir == Vector3.zero)
        {
            float absYawError = Mathf.Abs(Vector3.SignedAngle(tRoot.forward, targetLookDir, Vector3.up));

            if (shootSpotLocked) 
                targetStrideScale = absYawError > 8f ? 0.10f : 0.04f;
            else 
                targetStrideScale = absYawError > 15f && kickWaitTimer <= 0 ? 0.2f : 0f;
        }

        if (isKicking) 
            targetStrideScale = 0f;

        float absYaw = Mathf.Abs(Vector3.SignedAngle(tRoot.forward, targetLookDir, Vector3.up));
        if (absYaw > 45f)
        {
            float speedPenalty = Mathf.Clamp01(1f - (absYaw - 45f) / 90f);
            targetStrideScale *= Mathf.Max(speedPenalty, 0.2f);
        }

        float rampSpeed = (epStep < 100) ? 2.0f : 5.0f;
        if (targetStrideScale < smoothedStrideScale) 
            rampSpeed = 20.0f;
        if (isKicking) rampSpeed = 10.0f;
        smoothedStrideScale = Mathf.Lerp(smoothedStrideScale, targetStrideScale, rampSpeed * Time.fixedDeltaTime);

        float yawError = Vector3.SignedAngle(tRoot.forward, targetLookDir, Vector3.up);
        float turnCmd = Mathf.Clamp(yawError / 45f, -1f, 1f);
        if (isStrikingPhase) turnCmd = Mathf.Clamp(turnCmd, -0.15f, 0.15f);
        if (closeDynamicTurnRisk)
        {
            float closeTurnScale = Mathf.Lerp(0.25f, 1f, Mathf.InverseLerp(0.30f, 0.60f, distToBallFlat));
            turnCmd *= closeTurnScale;
        }

        float leftStrideScale = smoothedStrideScale * Mathf.Clamp01(1f + turnCmd * 0.4f);
        float rightStrideScale = smoothedStrideScale * Mathf.Clamp01(1f - turnCmd * 0.4f);

        for (int j = 0; j < 6; j++)
        {
            int li = sagJoints[j];
            if (legIdx[li] < 0)
                continue;

            bool isRight = (j < 3);

            float swing_uf = isRight ? uf2 : uf1;
            float stance_uf = isRight ? uf1 : uf2;
            float currentStride = isRight ? rightStrideScale : leftStrideScale;

            float targetDeg = 0f;
            float rawHip = 0f, rawKnee = 0f, rawAnkle = 0f;

            if (!isKicking)
            {
                rawHip = (18f * swing_uf - 12f * stance_uf) * currentStride * warmUp + stanceHipBias - (bodyPitch * 1.0f);
                rawHip = Mathf.Clamp(rawHip, -20f, 30f);

                float brakeFade = Mathf.Clamp01(Mathf.Abs(smoothedStrideScale) / 0.1f);
                float liftScale = Mathf.Max(Mathf.Abs(currentStride), 0.2f) * brakeFade;
                rawKnee = (30f * swing_uf + 5f * stance_uf) * liftScale * warmUp + gaitKneeBias;
                float baseAnkle = (rawKnee - rawHip) - bodyPitch;
                float swingLift = 15f * swing_uf * liftScale * warmUp;
                rawAnkle = baseAnkle + swingLift;
            }
            else
            {
                if (isRight)
                {
                    float kickProgress = (float)tq / T2;
                    if (kickProgress < 0.3f)
                    {
                        float w = kickProgress / 0.3f;
                        rawHip = stanceHipBias - 7f * w;
                        rawKnee = gaitKneeBias + 16f * w;
                    }
                    else if (kickProgress < 0.7f)
                    {
                        float w = (kickProgress - 0.3f) / 0.4f;
                        float strike = Mathf.Sin(w * Mathf.PI * 0.5f);
                    rawHip = Mathf.Lerp(stanceHipBias - 7f, 18f, strike);
                    rawKnee = Mathf.Lerp(gaitKneeBias + 16f, 2f, strike);
                    }
                    else
                    {
                        float w = (kickProgress - 0.7f) / 0.3f;
                        float retract = Mathf.Cos(w * Mathf.PI * 0.5f);
                    rawHip = Mathf.Lerp(stanceHipBias, 18f, retract);
                    rawKnee = Mathf.Lerp(gaitKneeBias, 2f, retract);
                    }
                    rawAnkle = (rawKnee - rawHip) - bodyPitch;
                    rawAnkle -= Mathf.Sin(kickProgress * Mathf.PI) * 6f;
                }
                else
                {
                    float kickProgress = (float)tq / T2;
                    float supportBlend = Mathf.Sin(kickProgress * Mathf.PI);
                    rawHip = Mathf.Lerp(stanceHipBias - bodyPitch, 4f - bodyPitch, supportBlend);
                    rawKnee = gaitKneeBias;
                    rawAnkle = (rawKnee - rawHip) - bodyPitch;
                }
                    
                rawHip *= warmUp;
                rawKnee *= warmUp;
                rawAnkle *= warmUp;
            }

            if (j == 0 || j == 3) // hip pitch
                targetDeg = rawHip * sagSigns[j];
            else if (j == 1 || j == 4) // knee
                targetDeg = rawKnee * sagSigns[j];
            else // ankle pitch
                targetDeg = rawAnkle * sagSigns[j];

            gaitTotal[legIdx[li]] = targetDeg;
        }

        float actionPenalty = 0f;
        float residualSqSum = 0f;
        int residualCount = 0;
        float kickControlBlend = isKicking ? Mathf.Sin(Mathf.Clamp01((float)tq / T2) * Mathf.PI) : 0f;
        float rightKickResidualScale = Mathf.Lerp(1f, 0.20f, kickControlBlend);
        float leftKickResidualScale = Mathf.Lerp(1f, 0.90f, kickControlBlend);
        float alignmentResidualScale = shootSpotLocked ? Mathf.Lerp(0.70f, 0.90f, Mathf.Clamp01(smoothedStrideScale / 0.25f)) : 1f;

        for (int i = 0; i < 12; i++)
        {
            // hip_roll(1,7), hip_yaw(2,8), ankle_roll(5,11) 
            if (i == 1 || i == 2 || i == 5 || i == 7 || i == 8 || i == 11)
                continue;
            if (legIdx[i] < 0)
                continue;
            int idx = legIdx[i];
            float residualScale = (i < 6 ? rightKickResidualScale : leftKickResidualScale) * alignmentResidualScale;
            float appliedResidual = (gaitU[idx] * actScale[i] * residualScale) * warmUp;
            float target = gaitTotal[idx] + appliedResidual;
            if (i == 3 || i == 9)
                target = Mathf.Max(target, 2f);
            SetDrive(revoluteJoints[idx], target, legGains[i], legDamps[i], legForces[i]);
            residualSqSum += appliedResidual * appliedResidual;
            residualCount++;

            float physicalDeg = gaitU[idx] * actScale[i];
            actionPenalty -= physicalDeg * physicalDeg * 0.000002f;
        }

        float freedomRamp = freedomRampSteps > 0f ? Mathf.Clamp01((float)(trainStepCounter / freedomRampSteps)) : 1f;
        if (!train || localPhase >= 3) 
            freedomRamp = 1f; 

        float yawScale = Mathf.Lerp(yawScaleMin, yawScaleMax, freedomRamp);
        float hipRollScale = Mathf.Lerp(hipRollScaleMin, hipRollScaleMax, freedomRamp);
        float ankleRollScale = Mathf.Lerp(ankleRollScaleMin, ankleRollScaleMax, freedomRamp);

        float rawYawR = legIdx[2] >= 0 ? gaitU[legIdx[2]] : 0f;
        float rawYawL = legIdx[8] >= 0 ? gaitU[legIdx[8]] : 0f;
        float yawDiff = (rawYawR - rawYawL) * 0.5f; 

        float kickYawBoost = 1f; 
        float yawTargetR = -yawDiff * yawScale * kickYawBoost * warmUp;
        float yawTargetL = yawDiff * yawScale * warmUp;
        float footStraighten = 0f;

        if (shootSpotLocked) footStraighten = 0.35f;
        if (isStrikingPhase) footStraighten = Mathf.Lerp(0.65f, 0.95f, Mathf.Clamp01((0.65f - distToBallFlat) / 0.35f));

        float rightFootStraighten = Mathf.Max(footStraighten, 0.95f * kickControlBlend);
        float leftFootStraighten = isKicking ? Mathf.Max(footStraighten, 0.70f) : footStraighten;

        yawTargetR = Mathf.Lerp(yawTargetR, 0f, rightFootStraighten);
        yawTargetL = Mathf.Lerp(yawTargetL, 0f, leftFootStraighten);
        if (legIdx[2] >= 0) SetDrive(revoluteJoints[legIdx[2]], yawTargetR, legGains[2], legDamps[2], legForces[2]);
        if (legIdx[8] >= 0) SetDrive(revoluteJoints[legIdx[8]], yawTargetL, legGains[8], legDamps[8], legForces[8]);

        float yawPhysDeg = yawDiff * yawScale;
        actionPenalty -= yawPhysDeg * yawPhysDeg * 0.000002f;

        float rawHipRollR = legIdx[1] >= 0 ? gaitU[legIdx[1]] : 0f;
        float rawHipRollL = legIdx[7] >= 0 ? gaitU[legIdx[7]] : 0f;
        float rawAnkRollR = legIdx[5] >= 0 ? gaitU[legIdx[5]] : 0f;
        float rawAnkRollL = legIdx[11] >= 0 ? gaitU[legIdx[11]] : 0f;

        float baseHipRollBias = 1.0f; 
        float baseAnkRollBias = 1.0f;

        float rightHipRollScale = Mathf.Lerp(1f, 0.45f, kickControlBlend);
        float leftHipRollScale = Mathf.Lerp(1f, 0.90f, kickControlBlend);
        float hipRollRLPartR = -rawHipRollR * hipRollScale * rightHipRollScale * warmUp;
        float hipRollRLPartL = rawHipRollL * hipRollScale * leftHipRollScale * warmUp;
        float ankRollRLPartR = rawAnkRollR * ankleRollScale * warmUp;
        float ankRollRLPartL = -rawAnkRollL * ankleRollScale * warmUp;
        
        float rightAnkleFreedom = 1f;
        if (isStrikingPhase && distToBallFlat < 0.65f) 
            rightAnkleFreedom = Mathf.Lerp(1f, 0.40f, Mathf.Clamp01((0.65f - distToBallFlat) / 0.35f));
        if (isKicking) 
            rightAnkleFreedom = Mathf.Lerp(rightAnkleFreedom, 0.10f, kickControlBlend);

        ankRollRLPartR *= rightAnkleFreedom;

        if (legIdx[2] >= 0) { residualSqSum += yawTargetR * yawTargetR; residualCount++; }
        if (legIdx[8] >= 0) { residualSqSum += yawTargetL * yawTargetL; residualCount++; }
        if (legIdx[1] >= 0) { residualSqSum += hipRollRLPartR * hipRollRLPartR; residualCount++; }
        if (legIdx[7] >= 0) { residualSqSum += hipRollRLPartL * hipRollRLPartL; residualCount++; }
        if (legIdx[5] >= 0) { residualSqSum += ankRollRLPartR * ankRollRLPartR; residualCount++; }
        if (legIdx[11] >= 0) { residualSqSum += ankRollRLPartL * ankRollRLPartL; residualCount++; }
        currentResidualRms = residualCount > 0 ? Mathf.Sqrt(residualSqSum / residualCount) : 0f;
        if (localPhase == 3)
        {
            p3ResidualSum += currentResidualRms;
            p3ResidualSamples++;
        }

        float hipRollTargetR = -baseHipRollBias + hipRollRLPartR;
        float hipRollTargetL = baseHipRollBias + hipRollRLPartL;
        float ankRollTargetR = baseAnkRollBias + ankRollRLPartR;
        float ankRollTargetL = -baseAnkRollBias + ankRollRLPartL;

        if (legIdx[1] >= 0) SetDrive(revoluteJoints[legIdx[1]], hipRollTargetR, legGains[1], legDamps[1], legForces[1]);
        if (legIdx[7] >= 0) SetDrive(revoluteJoints[legIdx[7]], hipRollTargetL, legGains[7], legDamps[7], legForces[7]);
        if (legIdx[5] >= 0) SetDrive(revoluteJoints[legIdx[5]], ankRollTargetR, legGains[5], legDamps[5], legForces[5]);
        if (legIdx[11] >= 0) SetDrive(revoluteJoints[legIdx[11]], ankRollTargetL, legGains[11], legDamps[11], legForces[11]);

        const float emaTau = 0.995f;
        hipRollEmaR = hipRollEmaR * emaTau + hipRollRLPartR * (1f - emaTau);
        hipRollEmaL = hipRollEmaL * emaTau + hipRollRLPartL * (1f - emaTau);
        ankleRollEmaR = ankleRollEmaR * emaTau + ankRollRLPartR * (1f - emaTau);
        ankleRollEmaL = ankleRollEmaL * emaTau + ankRollRLPartL * (1f - emaTau);

        actionPenalty -= rollDriftPenaltyCoef * (hipRollEmaR * hipRollEmaR + hipRollEmaL * hipRollEmaL + ankleRollEmaR * ankleRollEmaR + ankleRollEmaL * ankleRollEmaL);

        if (train)
        {
            Academy.Instance.StatsRecorder.Add("Soccer/FreedomRamp", freedomRamp);
            Academy.Instance.StatsRecorder.Add("Soccer/HipRollDriftR", hipRollEmaR);
            Academy.Instance.StatsRecorder.Add("Soccer/HipRollDriftL", hipRollEmaL);
            Academy.Instance.StatsRecorder.Add("Soccer/AnkleRollDriftR", ankleRollEmaR);
            Academy.Instance.StatsRecorder.Add("Soccer/AnkleRollDriftL", ankleRollEmaL);
        }

        // swing arm control
        float elbowBase = 70f;
        float elbowAmp = 10f;
        float shoulderBias = 10f;
        float shoulderAmp = 15f;
        float armOut = 5f;

        for (int i = 0; i < 15; i++)
            ud[i] = 0f;

        float currentShoulderAmp = shoulderAmp * (localPhase < 3 ? smoothedStrideScale : Mathf.Max(0.5f, smoothedStrideScale));
        float currentElbowAmp = elbowAmp * (localPhase < 3 ? smoothedStrideScale : Mathf.Max(0.5f, smoothedStrideScale));

        float rShoulderTarget = 0f, lShoulderTarget = 0f;
        float rElbowTarget = 0f, lElbowTarget = 0f;

        if (!isKicking)
        {
            rShoulderTarget = (uf2 - uf1) * currentShoulderAmp;
            lShoulderTarget = (uf1 - uf2) * currentShoulderAmp;
            rElbowTarget = (uf2 - uf1) * currentElbowAmp;
            lElbowTarget = (uf1 - uf2) * currentElbowAmp;
        }
        else
        {
            rShoulderTarget = 40f * uff;
            lShoulderTarget = -40f * uff;
            rElbowTarget = 20f * uff;
            lElbowTarget = -30f * uff;
        }

        ud[5] = shoulderBias + rShoulderTarget * warmUp;
        ud[10] = shoulderBias + lShoulderTarget * warmUp;
        ud[8] = elbowBase + rElbowTarget * warmUp;
        ud[13] = elbowBase + lElbowTarget * warmUp;
        ud[6] = -armOut; ud[11] = armOut;

        for (int i = 0; i < 15; i++)
            for (int j = 0; j < numJoints; j++)
                if (revoluteJoints[j].name == upN[i])
                {
                    SetDrive(revoluteJoints[j], ud[i], us[i], uDamp[i], uForce[i]);
                    break;
                }

        bool justSwitchedHalf = false;
        if (tp == 0 || tp == T1)
            justSwitchedHalf = true;
        int curGaitHalf = (tp < T1) ? 0 : 1;

        Transform swingFoot = (curGaitHalf == 0) ? leftFootTip : rightFootTip;
        Transform stanceFoot = (curGaitHalf == 0) ? rightFootTip : leftFootTip;

        if (swingFoot)
        {
            float h = swingFoot.position.y - groundY;
            if (h > maxSwingFootLift)
                maxSwingFootLift = h;
        }

        float stanceFootPenalty = 0f;
        if (stanceFoot)
        {
            float h = stanceFoot.position.y - groundY;
            if (h > 0.08f)
                stanceFootPenalty = -0.02f;
            else if (h > 0.02f)
                stanceFootPenalty = Mathf.Clamp(-h * 0.01f, -0.001f, 0f);
        }

        float tilt = Vector3.Angle(tRoot.up, Vector3.up);
        float absPitch = Mathf.Abs(Mathf.Asin(Mathf.Clamp(tRoot.forward.y, -1f, 1f)) * Mathf.Rad2Deg);
        float absRoll = Mathf.Abs(Mathf.Asin(Mathf.Clamp(tRoot.right.y, -1f, 1f)) * Mathf.Rad2Deg);
        if (fixbody && rootBody != null)
        {
            tilt = 0f;
            absPitch = 0f;
            absRoll = 0f;
        }

        targetVel = Vector3.Dot(rootBody.velocity, targetMoveDir);
        float speedMultiplier = Mathf.Clamp01(targetVel / currentSpeedThreshold);
        float postureReward = 0.01f * Mathf.Exp(-tilt * 0.07f) * (0.4f + 0.6f * speedMultiplier);

        float tiltError = Mathf.Max(0, tilt - 12.0f);
        float postureFactor = Mathf.Exp(-tiltError / 15.0f);

        float faceDot = Vector3.Dot(tRoot.forward, targetLookDir);
        float headingFactor = Mathf.Pow(Mathf.Clamp01(faceDot), 2f);

        Vector3 flatVel = new Vector3(rootBody.velocity.x, 0f, rootBody.velocity.z);
        float driftFactor = 1.0f;
        if (flatVel.magnitude > 0.2f && (localPhase == 1 || distToBallFlat > 1.5f))
        {
            float alignDot = Vector3.Dot(tRoot.forward, flatVel.normalized);
            driftFactor = Mathf.Pow(Mathf.Clamp01(alignDot), 2f);
        }

        float maxRewardedSpeed = Mathf.Max(currentSpeedThreshold * 1.5f, 0.3f);
        maxRewardedSpeed = Mathf.Min(maxRewardedSpeed, 0.6f);
        float velReward = Mathf.Clamp(targetVel, 0f, maxRewardedSpeed) * 0.09f * postureFactor * headingFactor * driftFactor;

        float latMulti = (localPhase == 1) ? 0.5f : 0.2f;
        Vector3 localVel = tRoot.InverseTransformDirection(rootBody.velocity);

        float swayError = Mathf.Max(0, Mathf.Abs(localVel.x) - 0.2f);
        float lateralPenalty = -0.01f * swayError * latMulti;

        float rollError = Mathf.Max(0, absRoll - 3.0f);
        float rollPenalty = -0.0005f * rollError;
        float comHeight = pos.y - groundY;
        float heightError = 0f;
        if (comHeight < 0.53f)
            heightError = 0.53f - comHeight;
        else if (comHeight > 0.85f)
            heightError = comHeight - 0.85f;
        float heightReward = -0.02f * heightError;

        float currentLatDist = Mathf.Abs(Vector3.Dot(pos - spawnPos, fieldRight));
        float pathDeviationPenalty = (localPhase == 1) ? -0.005f * currentLatDist * warmUp : 0f;

        float dribbleAlignmentPenalty = 0f;
        
        if (isStrikingPhase && !isKicking && distToBallFlat < 0.6f)
        {
            float currentAnkRollR = legIdx[5] >= 0 ? gaitU[legIdx[5]] : 0f;
            float currentAnkRollL = legIdx[11] >= 0 ? gaitU[legIdx[11]] : 0f;
            
            float ankleDistortion = Mathf.Abs(currentAnkRollR) + Mathf.Abs(currentAnkRollL);
            
            dribbleAlignmentPenalty = -0.005f * ankleDistortion;
        }

        float velocityRewardScale = localPhase >= 3 ? 0.20f : 1f;
        float stepReward = postureReward + velReward * velocityRewardScale + lateralPenalty + heightReward + stanceFootPenalty + rollPenalty + actionPenalty + pathDeviationPenalty + dribbleAlignmentPenalty;
        AddReward(stepReward);

        float maxTilt = (localPhase == 1) ? 30f : 20f;
        float maxPitch = (localPhase == 1) ? 30f : 22f;
        float maxRoll = (localPhase == 1) ? 30f : 25f;

        if (isKicking)
        {
            maxPitch = 30f;
            maxTilt = 30f;
        }

        if (tilt > maxTilt || absPitch > maxPitch || absRoll > maxRoll)
        {
            kickCompletionPending = false;
            episodeFell = true;
            if (localPhase == 3)
            {
                if (isKicking) p3FallKick++;
                else if (isStrikingPhase) p3FallStrike++;
                else if (shootSpotLocked) p3FallAlign++;
                else if (followBallActive) p3FallFollow++;
                else p3FallApproach++;
                p3FallSpeedSum += flatVel.magnitude;
                p3FallTiltSum += tilt;
                p3FallResidualSum += currentResidualRms;
                p3FallDynamicsSamples++;
            }
            AddReward(-1.0f);
            FinishEpisode();
            return;
        }

        if (kickCompletionPending)
        {
            if (!stableKickRewarded) 
                AddReward(0.25f);
            stableKickRewarded = true;
            kickCompletionPending = false;
        }

        int maxEp = 5000;
        if (localPhase == 1)
            maxEp = 3500;
        if (localPhase == 2)
            maxEp = 4500;
        if (epStep >= maxEp)
        {
            FinishEpisode();
            return;
        }

        if (transform.parent != null)
        {
            Vector3 offset = pos - spawnPos;
            float forwardDist = Vector3.Dot(offset, goalDir);
            float lateralDist = Mathf.Abs(Vector3.Dot(offset, fieldRight));

            if (localPhase == 1 && forwardDist > fieldLength - 0.5f)
            {
                float timeBonus = (maxEp - epStep) * 0.035f;
                AddReward(20f + timeBonus);
                FinishEpisode();
                return;
            }

            float agentBackLine = (localPhase == 1) ? -1.5f : -fieldHalfL;
            bool isAgentOutOfBounds = (forwardDist < agentBackLine - 0.3f || forwardDist > fieldLength + 0.3f || lateralDist > fieldHalfW + 0.3f );
            bool agentInGoalNet = (forwardDist > fieldLength - 0.2f) && (lateralDist < 2.5f);

            if (!agentInGoalNet && isAgentOutOfBounds)
            {
                AddReward(-1f);
                FinishEpisode();
                return;
            }
        }

        SoccerReward(tilt, maxEp);

        if (justSwitchedHalf)
        {
            if (maxSwingFootLift < 0.02f && epStep > 1)
                AddReward(-0.2f);
            maxSwingFootLift = 0f;
        }
    }

    void RewardShotOutcome(Vector3 ballVelocityFlat, bool fromKick)
    {
        bool isFirstShot = !episodeFirstShotEvaluated;
        if (isFirstShot) 
            episodeFirstShotEvaluated = true;
        
        if (ballVelocityFlat.sqrMagnitude < 0.0025f)
        {
            if (isFirstShot) 
                episodeFirstShotAccurate = false;
            AddReward(-0.2f);
            return;
        }

        Vector3 shotGoalDir = goalLeftPos - ball.position;
        shotGoalDir.y = 0f;
        if (shotGoalDir.sqrMagnitude < 0.0001f) return;
        shotGoalDir.Normalize();

        Vector3 shotRightDir = Vector3.Cross(Vector3.up, shotGoalDir).normalized;
        float forwardSpeed = Vector3.Dot(ballVelocityFlat, shotGoalDir);
        float lateralSpeed = Mathf.Abs(Vector3.Dot(ballVelocityFlat, shotRightDir));
        float shotAngle = Mathf.Atan2(lateralSpeed, Mathf.Max(0.01f, forwardSpeed)) * Mathf.Rad2Deg;
        float distToGoal = Vector3.Distance(ball.position, goalLeftPos);
        float safeAngle = Mathf.Clamp(Mathf.Atan2(1.1f, Mathf.Max(0.1f, distToGoal)) * Mathf.Rad2Deg, 3f, 30f);
        
        bool accurateShot = forwardSpeed > 0.25f && shotAngle <= safeAngle;
        if (isFirstShot) 
            episodeFirstShotAccurate = accurateShot;

        if (isFirstShot && accurateShot)
        {
            if (episodeFirstContactBodyAligned) AddReward(0.50f);
            if (episodeFirstContactFoot == 0) AddReward(0.75f);
            if (episodeFirstContactFoot == 0 && episodeFirstContactToeAligned) AddReward(0.25f);
        }
        
        if (forwardSpeed > 0f && shotAngle <= safeAngle)
        {
            float speedReward = Mathf.Clamp01(forwardSpeed / 1.5f);
            AddReward(0.5f + speedReward);
            if (fromKick) 
                AddReward(0.5f);
        }
        else if (forwardSpeed <= 0f)
        {
            AddReward(-1.5f);
        }
        else
        {
            AddReward(-0.5f);
        }
    }

    void SoccerReward(float tilt, int maxEp)
    {
        if (!ball || rootBody == null)
            return;
        var t = rootBody.transform;
        Vector2 flatPos = new Vector2(t.position.x, t.position.z);
        Vector2 flatBall = new Vector2(ball.position.x, ball.position.z);
        float flatD2B = Vector2.Distance(flatPos, flatBall);

        if (localPhase == 2)
        {
            if (flatD2B < BALL_R + 0.3f && !hasTouchedBall)
            {
                hasTouchedBall = true;
                float timeBonus = (maxEp - epStep) * 0.035f;
                AddReward(30f + timeBonus);
                FinishEpisode();
                return;
            }
            
            if (flatD2B < 2f)
                AddReward((2f - flatD2B) * 0.0002f);
        }

        Vector3 ballOffset = ball.position - spawnPos;
        float ballForwardDist = Vector3.Dot(ballOffset, goalDir);
        float ballLateralDist = Mathf.Abs(Vector3.Dot(ballOffset, fieldRight));

        if (localPhase >= 3)
        {
            Vector3 flatGoalDir = goalLeftPos - ball.position;
            flatGoalDir.y = 0f;
            if (flatGoalDir.sqrMagnitude > 0.0001f) 
                flatGoalDir.Normalize();

            Vector3 ballVelocityFlat = ballRb ? new Vector3(ballRb.velocity.x, 0f, ballRb.velocity.z) : Vector3.zero;

            if (shotContactCooldown > 0) shotContactCooldown--;

            if (shotEvaluationDelay > 0)
            {
                shotEvaluationDelay--;
                if (shotEvaluationDelay == 0) 
                    RewardShotOutcome(ballVelocityFlat, pendingShotWasKick);
            }

            float leftFootDistance = leftFootTip ? Vector3.Distance(leftFootTip.position, ball.position) : 10f;
            float rightFootDistance = rightFootTip ? Vector3.Distance(rightFootTip.position, ball.position) : 10f;
            Vector3 toeClosestPoint;
            Vector3 toeForwardDir;
            bool hasToeEdge = TryGetRightToeEdge(ball.position, out toeClosestPoint, out toeForwardDir);
            float rightContactDistance = hasToeEdge ? Vector3.Distance(toeClosestPoint, ball.position) : 10f;
            float rightContactLimit = BALL_R + 0.05f;
            Vector3 toeToBallNow = hasToeEdge ? ball.position - toeClosestPoint : Vector3.zero;
            toeToBallNow.y = 0f;
            Vector3 toeRightDir = hasToeEdge ? Vector3.Cross(Vector3.up, toeForwardDir).normalized : Vector3.zero;
            float toeForwardDistanceNow = hasToeEdge ? Vector3.Dot(toeToBallNow, toeForwardDir) : 0f;
            float toeSideErrorNow = hasToeEdge ? Mathf.Abs(Vector3.Dot(toeToBallNow, toeRightDir)) : 10f;

            Vector3 toeCenterOffsetNow = hasToeEdge ? ball.position - rightToeEdge.position : Vector3.zero;
            toeCenterOffsetNow.y = 0f;
            Vector3 toeEdgeDirNow = hasToeEdge ? rightToeEdge.right : Vector3.zero;
            toeEdgeDirNow.y = 0f;
            if (toeEdgeDirNow.sqrMagnitude > 0.0001f) toeEdgeDirNow.Normalize();
            float toeCenterErrorNow = hasToeEdge ? Mathf.Abs(Vector3.Dot(toeCenterOffsetNow, toeEdgeDirNow)) : 10f;

            Vector3 flatRightDir = Vector3.Cross(Vector3.up, flatGoalDir).normalized;
            Vector3 toeVelocityFlat = new Vector3(rightToeVelocity.x, 0f, rightToeVelocity.z);
            bool toeMotionAligned = rightToeVelocityReady && toeVelocityFlat.magnitude > 0.08f && Vector3.Angle(toeVelocityFlat, flatGoalDir) < 15f;
            float ballVelocityChange = (ballVelocityFlat - previousBallVelocity).magnitude;

            bool rightContact = hasToeEdge && rightContactDistance < rightContactLimit && toeForwardDistanceNow > 0.02f && toeForwardDistanceNow < rightContactLimit && toeSideErrorNow < 0.03f;
            bool rightFootNear = rightFootDistance < BALL_R + 0.20f;
            bool leftContact = leftFootDistance < BALL_R + 0.20f;
            bool actualContact = (rightContact || rightFootNear || leftContact) && ballVelocityChange > 0.12f && shotContactCooldown <= 0;
            bool rightToeFacingGoal = hasToeEdge && Vector3.Angle(toeForwardDir, flatGoalDir) < 15f;
            bool centeredRightContact = rightContact && toeCenterErrorNow < 0.020f;
            bool rightDominantContact = centeredRightContact && rightToeFacingGoal && toeMotionAligned && (!leftContact || rightContactDistance + 0.03f < leftFootDistance);
            
            if (actualContact)
            {
                if (localPhase == 3 && isKicking && !episodeKickContact)
                {
                    episodeKickContact = true;
                    p3KickContacts++;
                }
                if (!episodeFirstContactLogged)
                {
                    episodeFirstContactLogged = true;
                    episodeFirstContactState = contactStateBeforeNavigation;
                    bool bothFeetNear = rightFootNear && leftContact && Mathf.Abs(rightFootDistance - leftFootDistance) < 0.03f;
                    episodeFirstContactFoot = bothFeetNear ? 2 : ((rightContact || rightFootDistance < leftFootDistance) ? 0 : 1);
                    Vector3 contactForward = t.forward;
                    contactForward.y = 0f;
                    episodeFirstContactBodyAligned = contactForward.sqrMagnitude > 0.0001f && Vector3.Angle(contactForward, flatGoalDir) < 15f;
                    episodeFirstContactToeAligned = rightToeFacingGoal;
                    episodeFirstContactRightDominant = rightDominantContact;
                    episodeFirstContactBallMoving = previousBallVelocity.magnitude >= 0.12f;
                }

                if (!hasTouchedBall)
                {
                    hasTouchedBall = true;
                }

                pendingShotWasKick = isKicking;

                shotEvaluationDelay = 8;
                shotContactCooldown = 15;
            }

            float forwardSpeed = Vector3.Dot(ballVelocityFlat, flatGoalDir);
            float lateralSpeed = Mathf.Abs(Vector3.Dot(ballVelocityFlat, flatRightDir));
            AddReward(Mathf.Clamp(forwardSpeed - 0.5f * lateralSpeed, -1f, 1f) * 0.002f);
            previousBallVelocity = ballVelocityFlat;

            if (ballForwardDist > fieldLength && ballForwardDist < fieldLength + 0.6f && ballLateralDist < 1.5f && ball.position.y < initY + 1.8f)
            {
                goalCount++;
                Debug.Log($"GOAL! ({goalCount})");
                float timeBonus = (maxEp - epStep) * 0.035f;
                AddReward(60f + timeBonus);

                episodeScored = true;

                FinishEpisode();
                return;
            }
        }

        if (localPhase >= 2)
        {
            bool isBallOutOfBounds = (ballForwardDist < -fieldHalfL - BALL_R) || (ballForwardDist > fieldLength + BALL_R) || (ballLateralDist > fieldHalfW + BALL_R);
            if (isBallOutOfBounds)
            {
                AddReward(-0.5f);
                FinishEpisode();
                return;
            }
        }
    }

    Transform FindChildRecursive(Transform parent, string name)
    {
        foreach (Transform child in parent)
        {
            if (child.name == name)
                return child;
            var found = FindChildRecursive(child, name);
            if (found)
                return found;
        }
        return null;
    }

    void OnTriggerEnter(Collider c)
    {
        if (c.CompareTag("OutOfBounds"))
        {
            AddReward(-1f);
            FinishEpisode();
        }
    }
}
