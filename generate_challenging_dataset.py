"""
Generate a challenging dataset to test the enhanced pipeline.
This dataset includes:
- Edge cases and ambiguous items
- Heavy multilingual variation 
- Brand/model number mixing
- Abbreviations and acronyms
- Typos and misspellings
- Cross-category confusion items
"""
import pandas as pd
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_challenging_furniture_names(n=500):
    """Generate furniture names with maximum variation and edge cases."""
    
    # Base furniture types with extensive variations
    furniture_patterns = {
        'desk': [
            # English variations
            'desk', 'table', 'workstation', 'workspace', 'work table', 'work desk',
            'office desk', 'computer desk', 'writing desk', 'executive desk',
            'standing desk', 'sit-stand desk', 'height adjustable desk',
            
            # Multilingual
            'mesa', 'mesa de trabajo', 'escritorio', 'bureau', 'schreibtisch', 'biurko',
            'стол', 'デスク', '책상', 'مكتب', 'tavolo', 'skrivbord',
            
            # Typos and misspellings
            'deask', 'deks', 'tabel', 'tabl', 'workstaton', 'writting desk',
            
            # Brand/model mixing
            'IKEA desk', 'Herman Miller desk', 'Steelcase desk', 'HON table',
            'Desk Model D-100', 'Executive Table EX-200', 'Workstation WS-Pro'
        ],
        
        'chair': [
            # English variations
            'chair', 'seat', 'office chair', 'desk chair', 'task chair',
            'executive chair', 'ergonomic chair', 'swivel chair', 'armchair',
            'conference chair', 'meeting chair', 'mesh chair', 'leather chair',
            
            # Multilingual
            'silla', 'cadeira', 'chaise', 'stuhl', 'krzesło', 'стул',
            'イス', '의자', 'كرسي', 'sedia', 'stol',
            
            # Typos
            'chiar', 'char', 'seet', 'ofice chair', 'taask chair',
            
            # Brand/model
            'Aeron chair', 'Steelcase Leap', 'Herman Miller Sayl',
            'Chair Model C-300', 'Executive Seat EX-400'
        ],
        
        'cabinet': [
            # English variations
            'cabinet', 'filing cabinet', 'file cabinet', 'storage cabinet',
            'bookcase', 'shelf', 'shelving unit', 'storage unit', 'drawer',
            'cupboard', 'locker', 'wardrobe', 'armoire',
            
            # Multilingual  
            'armario', 'gabinete', 'archivador', 'armoire', 'schrank',
            'szafa', 'шкаф', 'キャビネット', '캐비넷', 'خزانة',
            
            # Typos
            'cabnet', 'cabnit', 'filing cabnet', 'stoarge cabinet',
            
            # Brand/model
            'HON filing cabinet', 'Steelcase storage', 'Cabinet ST-500'
        ],
        
        'sofa': [
            # English variations
            'sofa', 'couch', 'settee', 'loveseat', 'sectional', 'divan',
            'office sofa', 'reception sofa', 'lounge chair', 'bench',
            
            # Multilingual
            'sofá', 'canapé', 'divano', 'sofa', 'диван', 'ソファ', '소파',
            
            # Typos
            'soffa', 'counch', 'setee',
            
            # Brand/model
            'Reception Sofa RS-100', 'Lounge Seating LS-200'
        ]
    }
    
    # Edge cases and ambiguous items
    edge_cases = [
        # Ambiguous items that could be multiple categories
        'office furniture', 'workspace furniture', 'meeting room furniture',
        'furniture set', 'office set', 'desk accessories', 'office supplies',
        
        # Items with multiple functions
        'desk with storage', 'chair with desk', 'table chair combo',
        'storage desk unit', 'modular furniture', 'convertible furniture',
        
        # Very specific/technical terms
        'ergonomic workstation', 'height-adjustable sit-stand desk',
        'lumbar support task chair', 'lateral filing system',
        'modular shelving system', 'conference table extension',
        
        # Abbreviations
        'exec desk', 'conf table', 'task chr', 'fil cab', 'stor unit',
        'wrkstation', 'off chair', 'mtg table',
        
        # Numbers/codes that might confuse
        'Furniture Item 12345', 'Office Asset #67890', 'Desk Unit A1-B2',
        'Chair Type-X', 'Table v2.0', 'Cabinet Pro Max',
        
        # Mixed languages in single item
        'Mesa office desk', 'Silla executive chair', 'Bureau français',
        'Schreibtisch professional', 'Office стол', 'Business 机',
    ]
    
    names = []
    categories = []
    
    # Generate main furniture items
    for category, patterns in furniture_patterns.items():
        items_per_category = n // len(furniture_patterns)
        
        for _ in range(items_per_category):
            name = random.choice(patterns)
            
            # Add random modifiers 30% of the time
            if random.random() < 0.3:
                modifiers = ['black', 'white', 'brown', 'gray', 'wooden', 'metal', 
                           'plastic', 'leather', 'fabric', 'mesh', 'glass', 'steel',
                           'small', 'large', 'medium', 'standard', 'premium', 'basic',
                           'professional', 'executive', 'deluxe', 'compact']
                name = f"{random.choice(modifiers)} {name}"
            
            # Add random brand/model 20% of the time
            if random.random() < 0.2:
                brands = ['IKEA', 'Herman Miller', 'Steelcase', 'HON', 'Haworth', 
                         'Knoll', 'Teknion', 'Global', 'Kimball', 'Allsteel']
                name = f"{random.choice(brands)} {name}"
            
            # Add model number 15% of the time
            if random.random() < 0.15:
                model = f"Model {random.choice(['A', 'B', 'C', 'X', 'Pro'])}-{random.randint(100, 999)}"
                name = f"{name} {model}"
            
            names.append(name)
            categories.append('Furniture')
    
    # Add edge cases
    remaining = n - len(names)
    for _ in range(remaining):
        if edge_cases:
            name = random.choice(edge_cases)
            names.append(name)
            categories.append('Furniture')
    
    return list(zip(names, categories))

def generate_challenging_technology_names(n=200):
    """Generate technology names with heavy brand mixing and technical terms."""
    
    tech_patterns = {
        'computer': [
            # English variations
            'computer', 'PC', 'desktop', 'laptop', 'notebook', 'workstation',
            'all-in-one', 'mini PC', 'tower', 'desktop computer', 'personal computer',
            'business computer', 'gaming computer', 'thin client',
            
            # Multilingual
            'ordenador', 'computadora', 'ordinateur', 'computer', 'komputer',
            'компьютер', 'コンピュータ', '컴퓨터', 'حاسوب',
            
            # Brand-heavy (realistic for corporate environments)
            'Dell OptiPlex', 'HP EliteDesk', 'Lenovo ThinkCentre', 'Apple iMac',
            'Dell Latitude', 'HP ProBook', 'Lenovo ThinkPad', 'MacBook Pro',
            'Surface Laptop', 'Chromebook', 'Dell Precision', 'HP ZBook',
            
            # Technical specifications mixed in
            'Intel i7 Desktop', 'Core i5 Laptop', 'Ryzen Workstation',
            '16GB RAM Computer', '1TB SSD Laptop', '27-inch All-in-One',
            
            # Typos and abbreviations
            'lap top', 'note book', 'dekstop', 'compter', 'PC desktop',
            'i-Mac', 'think pad', 'surface pro', 'chrome book'
        ],
        
        'monitor': [
            # English variations
            'monitor', 'display', 'screen', 'LCD monitor', 'LED monitor',
            'curved monitor', 'ultrawide monitor', '4K monitor', 'widescreen',
            'dual monitor', 'secondary display', 'external monitor',
            
            # Multilingual
            'monitor', 'pantalla', 'écran', 'bildschirm', 'монитор',
            'モニター', '모니터', 'شاشة',
            
            # Brand/model heavy
            'Dell UltraSharp', 'LG UltraWide', 'Samsung Monitor', 'ASUS Monitor',
            'HP EliteDisplay', 'Acer Monitor', 'BenQ Monitor', 'ViewSonic',
            'Dell P2419H', 'LG 27UK850', 'Samsung C49RG90', 'ASUS PB278Q',
            
            # Technical specs
            '24-inch Monitor', '27-inch Display', '32-inch Screen', '4K Display',
            '1080p Monitor', '1440p Screen', 'FHD Monitor', 'QHD Display',
            
            # Typos
            'monitr', 'dsplay', 'scren', '4k monitor', 'ultra wide',
        ],
        
        'printer': [
            # English variations
            'printer', 'multifunction printer', 'MFP', 'all-in-one printer',
            'laser printer', 'inkjet printer', 'scanner', 'copier', 'fax machine',
            'photo printer', 'label printer', '3D printer',
            
            # Multilingual
            'impresora', 'imprimante', 'drucker', 'drukarka', 'принтер',
            'プリンター', '프린터', 'طابعة',
            
            # Brand/model
            'HP LaserJet', 'Canon PIXMA', 'Epson WorkForce', 'Brother MFC',
            'Xerox WorkCentre', 'Ricoh Aficio', 'Kyocera ECOSYS',
            'HP OfficeJet', 'Canon imageCLASS', 'Brother HL-L2350DW',
            
            # Abbreviations and tech terms
            'MFP device', 'AIO printer', 'B&W printer', 'Color laser',
            'Network printer', 'WiFi printer', 'USB printer',
            
            # Typos
            'printr', 'scaner', 'copyer', 'lazer printer', 'ink jet'
        ]
    }
    
    # Edge cases for technology
    tech_edge_cases = [
        # Ambiguous tech items
        'IT equipment', 'computer hardware', 'tech device', 'electronic device',
        'office electronics', 'digital device', 'computing device',
        
        # Accessories that might be confused
        'keyboard and mouse', 'mouse pad', 'USB hub', 'docking station',
        'power adapter', 'HDMI cable', 'ethernet cable', 'webcam',
        'speakers', 'headphones', 'microphone', 'document camera',
        
        # Mixed language tech terms
        'Ordinateur portable', 'Laptop computer', 'Desktop PC',
        'Imprimante laser', 'Monitor LCD', 'Écran LED',
        
        # Very technical/specific
        'Intel NUC mini PC', 'Microsoft Surface Studio', 'Apple Mac Pro',
        'NVIDIA Shield TV', 'Raspberry Pi 4', 'Arduino Uno',
        
        # Numbers and codes
        'Tech Asset #123', 'Computer ID-456', 'Device Type-A',
        'Hardware Unit 789', 'IT Equipment v3.1'
    ]
    
    names = []
    categories = []
    
    # Generate main tech items
    for category, patterns in tech_patterns.items():
        items_per_category = n // len(tech_patterns)
        
        for _ in range(items_per_category):
            name = random.choice(patterns)
            
            # Add random specs 40% of the time (tech items have more specs)
            if random.random() < 0.4:
                specs = ['wireless', 'bluetooth', 'USB-C', 'touchscreen', 'backlit',
                        'portable', 'refurbished', 'new', 'used', 'certified',
                        'business grade', 'enterprise', 'professional', 'gaming',
                        'energy efficient', 'quiet', 'compact', 'dual-band']
                name = f"{name} {random.choice(specs)}"
            
            # Add year/version 25% of the time
            if random.random() < 0.25:
                years = ['2020', '2021', '2022', '2023', '2024']
                versions = ['v2', 'v3', 'Gen 2', 'Gen 3', 'Pro', 'Plus', 'Max']
                modifier = random.choice(years + versions)
                name = f"{name} {modifier}"
            
            names.append(name)
            categories.append('Technology')
    
    # Add edge cases
    remaining = n - len(names)
    for _ in range(remaining):
        if tech_edge_cases:
            name = random.choice(tech_edge_cases)
            names.append(name)
            categories.append('Technology')
    
    return list(zip(names, categories))

def generate_challenging_service_names(n=300):
    """Generate service names with heavy acronyms and business terminology."""
    
    service_patterns = {
        'software': [
            # Common software
            'Microsoft Office', 'Office 365', 'Microsoft 365', 'MS Office',
            'Adobe Creative Suite', 'Adobe CC', 'Photoshop', 'AutoCAD',
            'Salesforce', 'SAP', 'Oracle', 'SQL Server', 'Windows license',
            'antivirus software', 'security software', 'backup software',
            
            # Multilingual software names
            'Software de oficina', 'Logiciel bureautique', 'Office Software',
            'Bürosoftware', 'ソフトウェア', '소프트웨어', 'برمجيات',
            
            # Specific versions and SKUs
            'Office 365 E3', 'Office 365 Business', 'Windows 10 Pro',
            'Adobe CC All Apps', 'AutoCAD 2024', 'Salesforce Enterprise',
            
            # Abbreviations and acronyms
            'CRM software', 'ERP system', 'BI tool', 'CAD software',
            'AV software', 'VPN client', 'RDP software', 'SSH client',
            
            # Typos
            'Ofice 365', 'Microsft Office', 'Adoby Creative', 'Saleforce',
            'Anti-virus', 'Anti virus', 'Back-up software'
        ],
        
        'subscription': [
            # Internet and telecom
            'internet service', 'broadband', 'fiber internet', 'DSL',
            'mobile plan', 'cell phone plan', 'data plan', 'unlimited plan',
            'business internet', 'dedicated line', 'VPN service',
            
            # Cloud services
            'cloud storage', 'cloud backup', 'AWS subscription', 'Azure subscription',
            'Google Workspace', 'Office 365 subscription', 'Dropbox Business',
            'OneDrive storage', 'iCloud storage', 'Box subscription',
            
            # Multilingual
            'servicio de internet', 'abonnement internet', 'Internet-Service',
            'インターネットサービス', '인터넷 서비스', 'خدمة الإنترنت',
            
            # Technical terms
            'SaaS subscription', 'PaaS service', 'IaaS platform',
            'CDN service', 'API subscription', 'SSL certificate',
            
            # Acronyms
            'ISP service', 'VoIP service', 'SIP trunk', 'PRI line',
            'MPLS connection', 'SD-WAN service'
        ],
        
        'support': [
            # IT support services
            'IT support', 'technical support', 'help desk', 'managed services',
            'maintenance contract', 'support contract', 'service agreement',
            'remote support', 'on-site support', 'phone support',
            
            # Multilingual
            'soporte técnico', 'support technique', 'technischer Support',
            'テクニカルサポート', '기술 지원', 'الدعم التقني',
            
            # Specific service types
            'hardware maintenance', 'software support', 'network support',
            'server maintenance', 'database support', 'security support',
            '24/7 support', 'business hours support', 'emergency support',
            
            # Acronyms
            'SLA agreement', 'MSP service', 'NOC service', 'SOC service',
            'ITIL support', 'ITSM service'
        ]
    }
    
    # Edge cases for services
    service_edge_cases = [
        # Ambiguous service items
        'IT services', 'business services', 'professional services',
        'consulting services', 'managed services', 'cloud services',
        
        # Mixed categories
        'software and support', 'hardware and maintenance', 'internet and phone',
        'cloud storage and backup', 'security software and support',
        
        # Very specific/technical
        'Microsoft Enterprise Agreement', 'Adobe VIP subscription',
        'AWS EC2 instances', 'Azure Active Directory Premium',
        'Salesforce Professional Edition', 'Oracle Database License',
        
        # Abbreviations and acronyms
        'SaaS platform', 'CRM system', 'ERP solution', 'BI platform',
        'API service', 'SDK license', 'IDE license', 'VM hosting',
        
        # Mixed language
        'Service de support', 'Servicio técnico', 'Support-Service',
        'クラウドサービス', '클라우드 서비스', 'خدمة سحابية',
        
        # Numbers and versions
        'Service Level 1', 'Support Tier 2', 'License v3.0',
        'Service Contract #456', 'Subscription ID-789'
    ]
    
    names = []
    categories = []
    
    # Generate main service items
    for category, patterns in service_patterns.items():
        items_per_category = n // len(service_patterns)
        
        for _ in range(items_per_category):
            name = random.choice(patterns)
            
            # Add service modifiers 35% of the time
            if random.random() < 0.35:
                modifiers = ['annual', 'monthly', 'enterprise', 'business', 'premium',
                           'standard', 'basic', 'professional', 'unlimited', 'limited',
                           'per user', 'per device', 'site license', 'volume license']
                name = f"{name} {random.choice(modifiers)}"
            
            # Add contract/license terms 20% of the time
            if random.random() < 0.2:
                terms = ['license', 'subscription', 'contract', 'agreement',
                        'service', 'plan', 'package', 'bundle']
                name = f"{name} {random.choice(terms)}"
            
            names.append(name)
            categories.append('Services')
    
    # Add edge cases
    remaining = n - len(names)
    for _ in range(remaining):
        if service_edge_cases:
            name = random.choice(service_edge_cases)
            names.append(name)
            categories.append('Services')
    
    return list(zip(names, categories))

def generate_challenging_dataset():
    """Generate the complete challenging dataset."""
    print("🎯 Generating challenging dataset with maximum variation...")
    
    # Generate each category
    furniture_data = generate_challenging_furniture_names(500)
    technology_data = generate_challenging_technology_names(200)
    services_data = generate_challenging_service_names(300)
    
    # Combine all data
    all_data = furniture_data + technology_data + services_data
    
    # Shuffle to mix categories
    random.shuffle(all_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data, columns=['product_name', 'true_category'])
    
    # Generate random barcodes (no correlation with category)
    df['barcode'] = [f"BC{random.randint(100000000, 999999999)}" for _ in range(len(df))]
    
    # Add some duplicate products with different names (realistic scenario)
    duplicates = []
    for _ in range(50):  # Add 50 duplicate products
        original_idx = random.randint(0, len(df) - 1)
        original_row = df.iloc[original_idx]
        
        # Create variation of the same product
        original_name = original_row['product_name']
        
        # Simple variations
        variations = [
            original_name.replace('desk', 'table'),
            original_name.replace('chair', 'seat'),
            original_name.replace('monitor', 'display'),
            original_name.replace('computer', 'PC'),
            original_name.upper(),
            original_name.lower(),
            f"Used {original_name}",
            f"{original_name} - Refurbished"
        ]
        
        new_name = random.choice([v for v in variations if v != original_name])
        
        duplicates.append({
            'product_name': new_name,
            'barcode': original_row['barcode'],  # Same barcode!
            'true_category': original_row['true_category']
        })
    
    # Add duplicates
    df_duplicates = pd.DataFrame(duplicates)
    df = pd.concat([df, df_duplicates], ignore_index=True)
    
    # Final shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save dataset
    output_file = "data/ultra_challenging_dataset.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Generated challenging dataset with {len(df)} items:")
    print(f"   📁 Saved to: {output_file}")
    print(f"   📊 Category distribution:")
    
    category_counts = df['true_category'].value_counts()
    for category, count in category_counts.items():
        print(f"      {category}: {count} items ({count/len(df)*100:.1f}%)")
    
    print(f"\n🎯 Challenge features included:")
    print(f"   ✓ Heavy multilingual variation (10+ languages)")
    print(f"   ✓ Brand/model number mixing")
    print(f"   ✓ Typos and misspellings") 
    print(f"   ✓ Abbreviations and acronyms")
    print(f"   ✓ Cross-category ambiguous items")
    print(f"   ✓ Technical specifications mixed in")
    print(f"   ✓ Duplicate products with different names")
    print(f"   ✓ Edge cases and unusual item descriptions")
    
    return df

if __name__ == "__main__":
    # Generate the challenging dataset
    df = generate_challenging_dataset()
    
    # Show some examples of challenging items
    print(f"\n📝 Examples of challenging items:")
    sample_items = df.sample(15)
    for _, row in sample_items.iterrows():
        print(f"   {row['true_category']:<12} | {row['product_name']}")
