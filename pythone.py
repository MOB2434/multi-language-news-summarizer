import json

product_name = input("Enter your name: ")
price = input("Enter the price: ")
discount = input("Enter the discount percentage: ")
vat = input("Enter the VAT percentage: ")

def calculate_final_price(price, discount, vat):
    discounted_price = price - (price * discount / 100)
    final_price = discounted_price + (discounted_price * vat / 100)
    return final_price

user_profile = {
    "name": product_name,
    "price": float(price),
    "discount": float(discount),
    "vat": float(vat),
    "final_price": calculate_final_price(float(price), float(discount), float(vat))
}

with open("user_profile.json", "w") as file:
    json.dump(user_profile, file, indent=4)

print("User profile saved to user_profile.json"



}

with open("data.json", "w") as f:
    json.dump(data, f, indent=4)ythone.py
    