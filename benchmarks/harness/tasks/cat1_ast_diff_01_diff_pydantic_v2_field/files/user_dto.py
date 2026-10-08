from pydantic import BaseModel


class UserDTO(BaseModel):
    tax_id: str
    name: str
