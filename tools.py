import json
import uuid
from typing import Dict, Optional
from datetime import datetime, timedelta
from langchain.agents import tool

from models import UserInfo, AppointmentInfo
from document_store import get_vectorstore

@tool
def search_documents(query: str) -> str:
    """
    Search through documents to find information relevant to the query.
    
    Args:
        query: The search query
    
    Returns:
        Relevant information from documents
    """
    # Get the vectorstore
    vectorstore = get_vectorstore()
    
    # Use the vectorstore for retrieval
    docs = vectorstore.similarity_search(query, k=3)
    
    if not docs:
        return "I couldn't find any relevant information in the documents."
    
    # Format the results
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}\n")
    
    return "\n".join(results)

@tool
def parse_date(date_text: str) -> str:
    """
    Parse various date formats including natural language like 'next Monday' 
    and return in YYYY-MM-DD format.
    
    Args:
        date_text: Text containing date information
    
    Returns:
        Date in YYYY-MM-DD format
    """

    today = datetime.now()
    
    days = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, 
            "friday": 4, "saturday": 5, "sunday": 6}
    
    if "next" in date_text.lower():
        for day, offset in days.items():
            if day in date_text.lower():
                current_day = today.weekday()
                days_ahead = offset - current_day
                if days_ahead <= 0:  
                    days_ahead += 7
                    
                target_date = today + timedelta(days=days_ahead)
                return target_date.strftime("%Y-%m-%d")
    
    if "tomorrow" in date_text.lower():
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "today" in date_text.lower():
        return today.strftime("%Y-%m-%d")
    elif "day after tomorrow" in date_text.lower():
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")
        
    date_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
        "%B %d, %Y", "%d %B, %Y", "%B %d %Y", "%d %B %Y"
    ]
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_text, fmt).date()
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    return "Could not parse date. Please use YYYY-MM-DD format."

@tool
def check_missing_appointment_fields(date: str = "", time: str = "", name: str = "", email: str = "", phone: str = "", purpose: str = "") -> str:
    """
    Checks which required fields for booking an appointment are missing.
    Returns a comma-separated list of missing fields.
    """
    missing = []
    if not date.strip():
        missing.append("date")
    if not time.strip():
        missing.append("time")
    if not name.strip():
        missing.append("name")
    if not email.strip():
        missing.append("email")
    if not phone.strip():
        missing.append("phone")
    if not purpose.strip():
        missing.append("purpose")
    return ", ".join(missing) if missing else "All fields present"


@tool
def book_appointment(input: str) -> str:
    """
    Book an appointment with the provided information.
    The input should be a JSON string with keys: date, time, name, email, phone, purpose.
    
    Example input:
    {
        "date": "2025-04-08",
        "time": "14:00",
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "1234567890",
        "purpose": "refund status"
    }
    """
    try:
        data: Dict = json.loads(input)

        date = data["date"]
        time = data["time"]
        name = data["name"]
        email = data["email"]
        phone = data["phone"]
        purpose = data["purpose"]

        user = UserInfo(
            name=name,
            email=email,
            phone=phone
        )

        appointment = AppointmentInfo(
            date=date,
            time=time,
            purpose=purpose,
            user_info=user
        )

        appointment_id = str(uuid.uuid4())[:8]
        #Call third party api for booking and save to database.
        return f"Appointment confirmed! Your appointment ID is {appointment_id}. You are scheduled for {date} at {time}. We'll send a confirmation to {email}."

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return f"Error booking appointment: {str(e)}"
