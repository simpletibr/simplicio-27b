def evens_of(nums: list[int]) -> list[int]:
    evens = []
    for x in nums:
        if x % 2 == 0:
            evens.append(x)
    return evens
