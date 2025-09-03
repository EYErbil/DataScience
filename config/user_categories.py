"""
Enhanced user-defined categories for ultra-challenging semantic analysis.
Edit this file to customize ONLY your main category names.
The enhanced system will learn from your data what belongs to each category.
"""

# Define ONLY your main category names here
# The enhanced system will learn from your data what items belong to each category
MAIN_CATEGORIES = [
    'Furniture',     # Office furniture: desks, chairs, tables, cabinets, etc.
    'Technology',    # IT equipment: computers, monitors, phones, tablets, etc.
    'Services'       # Digital services: software, internet, mobile plans, subscriptions, etc.
]

# Optional: Enhanced category descriptions for better zero-shot classification
CATEGORY_DESCRIPTIONS = {
    'Furniture': 'Office furniture, desks, chairs, tables, cabinets, storage, seating, workstations, and office furnishings',
    'Technology': 'Computers, laptops, monitors, printers, hardware, electronic devices, IT equipment, and tech accessories', 
    'Services': 'Software licenses, subscriptions, internet services, support contracts, cloud services, and IT services'
}

# You can add more categories as needed:
# 'Monitors',
# 'Kitchen_Items', 
# 'Storage_Furniture',
# 'Lighting',
# 'Electronics'

# ✨ The magic: You only provide category names above ✨
# The AI will analyze your inventory data and learn:
# - What words indicate "Tables" (desk, mesa, masa, writing surface, etc.)
# - What words indicate "Chairs" (sandalye, silla, seat, ergonomic, etc.) 
# - What words indicate "Computers" (PC, bilgisayar, ordenador, laptop, etc.)
# - And so on for each category you define!

# No more manually defining keywords - the system learns from YOUR data!
