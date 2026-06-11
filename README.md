# X For You Algorithm — Formula Artifacts & Strategy

## Strategy (derived from the code below)

- **Maximize author-engaged replies.** `reply_engaged_by_author` is the highest-weighted head (published 2023 weight: 75.0 vs 0.5 for a like, ~150×). Post content that invites replies; reply back within the scoring window.
- **Win the first 6 hours.** Relevance score is multiplied by a sigmoid age decay with halflife = 360 min (A2). Real-time engagement counters decay with a 30-min half-life; SimClusters tweet embeddings with an 8-h half-life (A6). Post at peak audience time.
- **Early likes drive distribution, not just rank.** Likes are the retrieval key into UTEG (liked-by-followed-accounts) and SimClusters (out-of-network interest clusters, top ~400 tweets/cluster). 1 like from a followed account makes the tweet a candidate for that follower's graph within seconds.
- **Tweet lifetime ≈ 24–48 h hard cap.** UTEG drops candidates > 24 h (A7); in-network Earlybird window ≈ 48 h.
- **Never trigger negative heads.** `ReportParam` range is [-20000, 0] (negative-only); published weights: report = −369, negative feedback = −74. One report ≈ −738 likes. "Show less often" applies a per-viewer ×0.2 multiplier for up to 140 days (A5).
- **Avoid bare links on low-reputation accounts.** Non-media/non-news link + tweepcred < 25 → spam-scored, unless the tweet already has ≥ 1 engagement (A4).
- **Text quality:** length and entropy dominate the light-ranker text score (0.5 + 0.25 of weight); offensive terms apply ×0.2 damping; ALL-CAPS ("shout") and multi-hashtag flags are penalized (A3).
- **Don't self-cannibalize.** Per-feed author diversity decay: 2nd tweet ×0.5, floor 0.25 (A5). Replies and out-of-network impressions ×0.75 (A5). Space posts out.

---

## A1. Final score: weighted sum of predicted engagement probabilities

`home-mixer/server/src/main/scala/com/twitter/home_mixer/util/RerankerUtil.scala:91`

```scala
def aggregateWeightedScores(
  query: PipelineQuery,
  scoresAndWeights: Seq[(Double, Double, Double)],  // (score, maxHeadScore, weight)
  negativeFilterCounter: Counter
): Double = {
  ...
  val combinedScoreSum: Double = {
    scoresAndWeights.foldLeft(0.0) {
      case (combinedScore, (score, maxHeadScore, weight)) =>
        if (weight >= 0.0 || useWeightForNeg) {
          combinedScore + score * weight                 // Σ p(engagement) × w
        } else {
          // Apply filtering logic only for negative weights
          val normScore = if (maxHeadScore == 0.0) 0.0 else score / maxHeadScore
          val negFilterNorm = enableNegNormalized && normScore > thresholdNegativeNormalized
          val negFilterConstant = enableNegConstant && score > thresholdNegative
          if (negFilterNorm || negFilterConstant) {
            ...
            combinedScore + weight                       // negative head penalty
          } else combinedScore
        }
    }
  }
```

Predicted heads (`home-mixer/.../model/PredictedScoreFeature.scala`): favorite, retweet, reply, reply_engaged_by_author, good_click_v1/v2 (conversation + dwell ≥ 2 min), good_profile_click, video_quality_view, video_quality_view_immersive, video_watch_time_ms, bookmark, share, share_menu_click, dwell, tweet_detail_dwell, profile_dwelled, open_link, screenshot, negative_feedback_v2, report.

Weights in this snapshot default to 0.0 (runtime-configured), ranges in `HomeGlobalParams.scala:786-1028`; `ReportParam` ∈ [-20000, 0], `Weak/StrongNegativeFeedbackParam` ∈ [-1000, 0]. Published 2023 production values (companion repo `the-algorithm-ml/projects/home/recap`):

| head | weight | like-equivalents |
|---|---|---|
| reply_engaged_by_author | 75.0 | 150 |
| reply | 13.5 | 27 |
| good_profile_click | 12.0 | 24 |
| good_click (conv + dwell≥2min) | ~11.0 | 22 |
| retweet | 1.0 | 2 |
| favorite | 0.5 | 1 |
| video_playback_50 | 0.005 | 0.01 |
| negative_feedback | −74.0 | −148 |
| report | −369.0 | −738 |

## A2. Age decay (Earlybird relevance, multiplicative)

`src/thrift/com/twitter/search/common/ranking/ranking.thrift:14`

```thrift
struct ThriftAgeDecayRankingParams {
  // the rate in which the score of older tweets decreases
  1: optional double slope = 0.003
  // the age, in minutes, where the age score of a tweet is half of the latest tweet
  2: optional double halflife = 360.0
  // the minimal age decay score a tweet will have
  3: optional double base = 0.6
}
```

`src/java/com/twitter/search/common/relevance/features/AgeDecay.java:64`

```java
public static double compute(
    double base, double maxBoost, double halflife, double slope, double age) {
  return base + ((maxBoost - base) / (1 + Math.exp(slope * (age - halflife))));
}
// SLOPE_COEFF = 4.0; slope = 4.0 / halflife
```

decay(age) = 0.6 + 0.4 / (1 + e^{(4/360)·(age_min − 360)}); decay(0) ≈ 1.0, decay(6 h) = 0.8, decay(∞) → 0.6.

## A3. Light-ranker text quality score

`src/java/com/twitter/search/common/relevance/scorers/TweetTextScorer.java:27`

```java
// text_score = offensive_text_damping * offensive_username_damping *
//   (length_weight·length + readability_weight·readability +
//    shout_weight·shout + entropy_weight·entropy + link_weight·link)
DEFAULT_OFFENSIVE_TERM_DAMPING = 0.2d;
DEFAULT_OFFENSIVE_NAME_DAMPING = 0.2d;
DEFAULT_LENGTH_WEIGHT      = 0.5d;
DEFAULT_READABILITY_WEIGHT = 0.1d;
DEFAULT_SHOUT_WEIGHT       = 0.1d;   // ALL-CAPS penalty
DEFAULT_ENTROPY_WEIGHT     = 0.25d;
DEFAULT_LINK_WEIGHT        = 0.05d;
```

## A4. Link-spam gate (tweepcred threshold)

`src/java/com/twitter/search/earlybird/search/relevance/scoring/SpamVectorScoringFunction.java`

```java
private static final int MIN_TWEEPCRED_WITH_LINK = 25;   // PageRank author reputation
private static final int ENGAGEMENTS_NO_FILTER = 1;
static final float NOT_SPAM_SCORE =  0.5f;
static final float SPAM_SCORE     = -0.5f;
...
if (retweetCount + replyCount + favoriteCount >= ENGAGEMENTS_NO_FILTER) {
  return NOT_SPAM_SCORE;   // any engagement bypasses the link-spam filter
}
return SPAM_SCORE;
```

## A5. Post-model multiplicative rescorers

`home-mixer/.../scorer/RescoringFactorProvider.scala`

```scala
object RescoreOutOfNetwork extends RescoringFactorProvider {
  def selector(...) = !candidate.features.getOrElse(InNetworkFeature, false)
  def factor(...)   = query.params(OutOfNetworkScaleFactorParam)   // default 0.75
}
object RescoreReplies extends RescoringFactorProvider {
  def selector(...) = candidate.features.getOrElse(InReplyToTweetIdFeature, None).isDefined
  def factor(...)   = query.params(ReplyScaleFactorParam)          // default 0.75
}
```

Author/source diversity decay — `AuthorBasedListwiseRescoringProvider.scala:54` (same formula in `ImpressedAuthorDecayRescoringProvider`, `CandidateSourceDiversityListwiseRescoringProvider`):

```scala
): Double = (1 - floor) * Math.pow(decayFactor, index) + floor
// author:  decayFactor = 0.5, floor = 0.25  → 1.0, 0.625, 0.4375, ... ≥ 0.25
// source:  decayFactor = 0.9, floor = 0.8
```

Feedback fatigue ("show less often") — `FeedbackFatigueScorer.scala:38`:

```scala
val DurationForDiscounting = 140.days
private val ScoreMultiplierLowerBound = 0.2
private val ScoreMultiplierUpperBound = 1.0
private val ScoreMultiplierIncrementsCount = 4
// per-viewer multiplier steps 0.2 → 1.0 over 140 days post-feedback
```

## A6. Engagement-signal time decay (retrieval side)

SimClusters tweet-embedding half-life — `src/scala/com/twitter/simclusters_v2/scio/bq_generation/simclusters_index_generation/Config.scala:55`:

```scala
val tweetEmbeddingsHalfLife: Int = 28800000   // 8 hours in ms
```

Decay application — `.../bq_generation/common/IndexGenerationUtil.scala:51`:

```scala
val scaledTime =
  SnowflakeId.unixTimeMillisFromId(tweetId) * math.log(2.0) / tweetEmbeddingsHalfLife
val decayedValue = DecayedValue(tweetScore, scaledTime)   // score × 2^(−age/halfLife)
```

Real-time ranking features use 30-min half-life tweet-engagement aggregates (`TimelinesOnlineAggregationConfigBase.scala`), surfaced to the ranker as `timelines.earlybird.decayed_{favorite,retweet,reply,quote}_count`.

## A7. Hard candidate-age windows

```scala
// UTEG (tweets liked by your follow graph) — recos/user_tweet_entity_graph/RecosConfig.scala:33
val maxTweetAgeInMillis: Long = 24 * 60 * 60 * 1000        // 24 h

// SimClusters ANN — cr-mixer/.../SimClustersANNConfig.scala
// default 24.hours; real-time variant 1.hour; evergreen 48.hours
```

In-network Earlybird fetch window ≈ 48 h; candidate mix per request ≈ 600 in-network / 400 tweet-mixer (SimClusters+TwHIN) / 300 UTEG / 100 FRS (`ScoredTweetsParam.scala`).
