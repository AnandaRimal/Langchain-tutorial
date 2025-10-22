from pydantic import BaseModel, ValidationError

# Step 1: Define a Pydantic model
class User(BaseModel):
    name: str
    age: int
    email: str

# Step 2: Function to create and validate user data
def create_user(data: dict) -> User | None:
    # Pydantic automatically validates data
    result = User.model_validate(data)  # if invalid, it raises ValidationError
    return result

# Step 3: Valid data
data1 = {"name": "Ananda", "age": 22, "email": "ananda@example.com"}
user = create_user(data1)
print(user)

# Step 4: Invalid data (this will raise ValidationError automatically)
data2 = {"name": "Ram", "age": "twenty", "email": "ram@example.com"}
user2 = create_user(data2)
print(user2)