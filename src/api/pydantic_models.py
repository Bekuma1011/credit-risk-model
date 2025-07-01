from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    TransactionId: Optional[str]
    BatchId: Optional[str]
    AccountId: Optional[str]
    SubscriptionId: Optional[str]
    CustomerId: str
    CurrencyCode: Optional[str]
    CountryCode: Optional[int]
    ProviderId: Optional[str]
    ProductId: Optional[str]
    ProductCategory: Optional[str]
    ChannelId: Optional[str]
    Amount: float
    Value: int
    TransactionStartTime: str  #
    PricingStrategy: Optional[int]
    FraudResult: Optional[int]

class PredictionResponse(BaseModel):
    risk_probability: float