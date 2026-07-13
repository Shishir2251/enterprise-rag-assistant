from pydantic import BaseModel, ConfigDict


class DocumentChunkResponse(BaseModel):
    chunk_index: int
    content: str
    character_count: int
    page_number: int | None

    model_config = ConfigDict(from_attributes=True)
