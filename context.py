import uuid 
from dataclasses import dataclass

@dataclass
class RequestContext:
    session_id: str
    request_id: str

    
    @classmethod 
    def new(cls, session_id: str = None) -> "RequestContext":
        return cls(
            session_id=session_id or str(uuid.uuid4()),
            request_id=str(uuid.uuid4())
        )