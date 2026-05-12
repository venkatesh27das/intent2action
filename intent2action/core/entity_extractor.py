"""Entity extraction step."""

from intent2action.schemas.entity import ExtractedEntity


class EntityExtractor:
    """Coerce model entities into schema objects."""

    def extract(self, raw_entities: list[dict]) -> list[ExtractedEntity]:
        """Validate extracted entity dictionaries."""

        entities: list[ExtractedEntity] = []
        for item in raw_entities:
            try:
                entities.append(ExtractedEntity.model_validate(item))
            except Exception:
                continue
        return entities

