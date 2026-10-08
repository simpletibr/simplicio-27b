def cart_total(cart: list) -> float:
    total = sum([item.price for item in cart])
    return total
