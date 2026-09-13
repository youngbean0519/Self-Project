NEGATIVE_CLUSTERS = {
    "hostility_anger": [
        "불평/불만",
        "화남/분노",
        "역겨움/징그러움",
        "짜증",
        "증오/혐오",
        "한심함"
    ],

    "sadness": [
        "슬픔",
        "서러움",
    ],

    "boredom_tiredness": [
        "지긋지긋",
        "재미없음",
    ],

    "exhaustion_avoidance": [
        "힘듦/지침",
        "귀찮음",
        "부담/안_내킴",
    ],

    "fear_anxiety": [
        "의심/불신",
        "공포/무서움",
        "불안/걱정",
    ],

    "disappointment_frustration": [
        "안타까움/실망",
        "절망",
        "패배/자기혐오",
    ],

    "self_reflection": [
        "부끄러움",
        "죄책감",
    ],
}

POSITIVE_CLUSTERS = {
    "gratitude_respect": [
        "고마움",
        "존경",
    ],

    "affection": [
        "환영/호의",
        "아껴주는",
    ],

    "satisfaction_stability": [
        "뿌듯함",
        "편안/쾌적",
        "흐뭇함(귀여움/예쁨)",
        "안심/신뢰",
    ],

    "happiness": [
        "즐거움/신남",
        "행복",
        "기쁨",
        "감동/감탄",
    ],
}

SURPRISE_CLUSTER = {
    "surprise_confusion": [
        "당황/난처",
        "경악",
        "놀람",
        "어이없음",
    ]
}

UNCLUSTERED_LABELS = [
    "기대감",
    "우쭐댐/무시함",
    "비장함",
    "신기함/관심",
    "깨달음",
    "불쌍함/연민",
]

SPECIAL_LABELS = [
    "없음",
]

LABEL_CLUSTERS = {
    **NEGATIVE_CLUSTERS,
    **POSITIVE_CLUSTERS,
    **SURPRISE_CLUSTER,
}