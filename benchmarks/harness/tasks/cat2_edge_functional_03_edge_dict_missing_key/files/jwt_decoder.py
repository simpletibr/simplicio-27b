def get_user_role(claims: dict) -> str:
    return claims['user']['role']
