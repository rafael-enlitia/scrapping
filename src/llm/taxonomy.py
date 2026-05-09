"""Fixed topic taxonomy shared by prompts, classification, and dashboard."""

from enum import Enum


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class Topic(str, Enum):
    PERFORMANCE = "performance"
    UI_UX = "ui_ux"
    BUGS = "bugs"
    FEATURES = "features"
    PRICING = "pricing"
    PRIVACY_SECURITY = "privacy_security"
    CUSTOMER_SUPPORT = "customer_support"
    UPDATES = "updates"
    USABILITY = "usability"
    OTHER = "other"


SENTIMENT_VALUES = [s.value for s in Sentiment]
TOPIC_VALUES = [t.value for t in Topic]

TOPIC_DESCRIPTIONS = {
    Topic.PERFORMANCE: "Speed, lag, battery drain, crashes",
    Topic.UI_UX: "Design, layout, visual appearance, navigation",
    Topic.BUGS: "Errors, glitches, broken features",
    Topic.FEATURES: "Missing features, feature requests, feature praise",
    Topic.PRICING: "Cost, subscriptions, in-app purchases, ads",
    Topic.PRIVACY_SECURITY: "Data privacy, permissions, security concerns",
    Topic.CUSTOMER_SUPPORT: "Support responsiveness, help quality",
    Topic.UPDATES: "Effects of recent updates, version changes",
    Topic.USABILITY: "Ease of use, learning curve, accessibility",
    Topic.OTHER: "Topics that don't fit above categories",
}
