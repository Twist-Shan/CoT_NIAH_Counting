from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityDomain:
    """Lexical specification for the audit-record counting task.

    ``legacy_record_key`` remains ``city`` for every domain because the V5
    trace parser historically uses that field name for the entity registry.
    The value itself is an arbitrary entity label; callers should use
    ``name``/``singular`` when presenting the experiment.
    """

    name: str
    singular: str
    plural: str
    entity_names: tuple[str, ...]
    legacy_record_key: str = "city"

    @property
    def indefinite_article(self) -> str:
        return (
            "An"
            if self.singular[:1].lower() in {"a", "e", "i", "o", "u"}
            else "A"
        )

    @property
    def counting_cue(self) -> str:
        return (
            f"You will need to count all {self.singular}-score audit records "
            "in the passage below.\n"
            f"{self.indefinite_article} {self.singular}-score audit record "
            f"names one {self.singular} "
            f"and gives that {self.singular}'s numeric score."
        )

    @property
    def count_question(self) -> str:
        return (
            f"How many {self.singular}-score audit records are in the passage?"
        )

    def record_sentence(self, entity: str, score: int) -> str:
        return (
            f"In the 2024 {self.singular} score audit, {entity} received a "
            f"score of {int(score)}."
        )


_FLOWERS = (
    "Rose",
    "Tulip",
    "Daisy",
    "Orchid",
    "Sunflower",
    "Lavender",
    "Jasmine",
    "Marigold",
    "Peony",
    "Iris",
    "Lily",
    "Carnation",
    "Violet",
    "Daffodil",
    "Camellia",
    "Magnolia",
    "Poppy",
    "Dahlia",
    "Azalea",
    "Hibiscus",
    "Begonia",
    "Gardenia",
    "Primrose",
    "Zinnia",
    "Petunia",
    "Anemone",
    "Freesia",
    "Hyacinth",
    "Narcissus",
    "Verbena",
)

_ANIMALS = (
    "Otter",
    "Badger",
    "Falcon",
    "Rabbit",
    "Leopard",
    "Dolphin",
    "Penguin",
    "Giraffe",
    "Beaver",
    "Gazelle",
    "Heron",
    "Meerkat",
    "Panther",
    "Walrus",
    "Yak",
    "Fox",
    "Koala",
    "Llama",
    "Raven",
    "Turtle",
    "Bison",
    "Cobra",
    "Ferret",
    "Gecko",
    "Hamster",
    "Iguana",
    "Jaguar",
    "Lemur",
    "Marten",
    "Narwhal",
)


ENTITY_DOMAINS = {
    "city": EntityDomain("city", "city", "cities", ()),
    "flower": EntityDomain("flower", "flower", "flowers", _FLOWERS),
    "animal": EntityDomain("animal", "animal", "animals", _ANIMALS),
}


def resolve_entity_domain(value: str | None) -> EntityDomain:
    name = "city" if value is None else str(value).strip().lower()
    try:
        return ENTITY_DOMAINS[name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported entity domain {value!r}; expected one of "
            f"{sorted(ENTITY_DOMAINS)}"
        ) from error


def native_user_text(passage: str, *, entity_domain: str = "city") -> str:
    domain = resolve_entity_domain(entity_domain)
    return (
        f"{domain.counting_cue}\n\n"
        f"<passage>\n{str(passage)}\n</passage>\n\n"
        f"{domain.count_question}\n"
        "Reason concisely without repeating or restarting.\n"
        "Stop as soon as you determine the count, then output exactly one line:\n"
        "Total: <integer>"
    )


def nonthinking_query_text(
    *, entity_domain: str = "city", answer_format: str = "numeric"
) -> str:
    domain = resolve_entity_domain(entity_domain)
    if answer_format == "numeric":
        return (
            f"{domain.count_question}\n"
            "Do not explain, reason aloud, quote, or list any records.\n"
            "Write the count using ordinary decimal digits, with no space after "
            "the colon.\n"
            "Your entire response must be exactly one line:\n"
            "Total:<integer>"
        )
    if answer_format == "number_word":
        return (
            f"{domain.count_question}\n"
            "Do not explain, reason aloud, quote, or list any records.\n"
            "Write the count as one lowercase English number word from one "
            "through ten,\n"
            "with no space after the colon.\n"
            "Your entire response must be exactly one line:\n"
            "Total:<number word>"
        )
    raise ValueError(f"Unsupported answer format: {answer_format!r}")
