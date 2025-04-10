from pydantic import BaseModel, EmailStr, field_validator

from pydantic import BaseModel, field_validator
import re

class UserInfo(BaseModel):
    name: str
    email: EmailStr
    phone: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty.")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v):
        phone_regex = r"^\d{10}$"
        if not re.match(phone_regex, v):
            raise ValueError("Invalid phone number. Must be exactly 10 digits.")
        return v


class AppointmentInfo(BaseModel):
    date: str
    time: str
    purpose: str
    user_info: UserInfo

    @field_validator("date", "time", "purpose")
    @classmethod
    def fields_not_empty(cls, v, info):
        if not v.strip():
            raise ValueError(f"{info.field_name.capitalize()} cannot be empty.")
        return v
