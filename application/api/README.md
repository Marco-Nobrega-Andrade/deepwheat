## Setup

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip3 install -r requirements.txt
```

### Postman collection
File: postman_collection/deep_wheat_collection.postman_collection.json
Steps to Use
Open Postman

Click Import → Select the collection file

Use the Login request to authenticate

The login request automatically stores the JWT in {{token}}

Other requests use Bearer {{token}} for authentication