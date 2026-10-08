from fastapi import FastAPI

app = FastAPI()


@app.get('/v1/users')
def list_users():
    return []
