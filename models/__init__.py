class BaseModel:
    def __init__(self) -> None:
        ...

    def forward(self):
        raise NotImplementedError
    
    def