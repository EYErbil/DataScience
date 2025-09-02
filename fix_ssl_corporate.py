#!/usr/bin/env python3
"""
Aggressive SSL bypass for corporate networks.
Run this BEFORE starting your Jupyter notebook to fix HuggingFace downloads.
"""

import os
import ssl
import warnings

def setup_aggressive_ssl_bypass():
    """Setup comprehensive SSL bypass for corporate firewalls."""
    
    print("🔓 Setting up aggressive SSL bypass for corporate networks...")
    
    # 1. Environment variables
    ssl_env_vars = {
        'CURL_CA_BUNDLE': '',
        'REQUESTS_CA_BUNDLE': '',
        'SSL_VERIFY': 'false',
        'PYTHONHTTPSVERIFY': '0',
        'TRANSFORMERS_OFFLINE': '0',
        'HF_HUB_DISABLE_TELEMETRY': '1',
        'HF_HUB_OFFLINE': '0',
        'REQUESTS_CA_BUNDLE': '',
        'CURL_CA_BUNDLE': '',
    }
    
    for key, value in ssl_env_vars.items():
        os.environ[key] = value
    
    print("✅ Environment variables set")
    
    # 2. SSL context bypass
    ssl._create_default_https_context = ssl._create_unverified_context
    print("✅ SSL context patched")
    
    # 3. Urllib3 warnings
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        print("✅ SSL warnings disabled")
    except ImportError:
        print("⚠️ urllib3 not available")
    
    # 4. Patch requests globally
    try:
        import requests
        
        # Patch session requests
        original_request = requests.Session.request
        def patched_request(self, *args, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = kwargs.get('timeout', 30)
            return original_request(self, *args, **kwargs)
        requests.Session.request = patched_request
        
        # Patch module-level functions
        original_get = requests.get
        original_post = requests.post
        original_put = requests.put
        original_patch = requests.patch
        original_delete = requests.delete
        
        def patched_get(*args, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = kwargs.get('timeout', 30)
            return original_get(*args, **kwargs)
            
        def patched_post(*args, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = kwargs.get('timeout', 30)
            return original_post(*args, **kwargs)
            
        def patched_put(*args, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = kwargs.get('timeout', 30)
            return original_put(*args, **kwargs)
            
        def patched_patch(*args, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = kwargs.get('timeout', 30)
            return original_patch(*args, **kwargs)
            
        def patched_delete(*args, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = kwargs.get('timeout', 30)
            return original_delete(*args, **kwargs)
        
        requests.get = patched_get
        requests.post = patched_post
        requests.put = patched_put
        requests.patch = patched_patch
        requests.delete = patched_delete
        
        print("✅ Requests library patched")
        
    except ImportError:
        print("⚠️ Requests not available")
    
    # 5. Patch urllib3
    try:
        import urllib3.util.ssl_
        urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL'
        urllib3.util.ssl_.DEFAULT_CIPHERS += ':!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA'
        print("✅ Urllib3 SSL patched")
    except Exception:
        print("⚠️ Urllib3 SSL patch failed")
    
    # 6. Patch HuggingFace Hub
    try:
        import huggingface_hub
        
        # Patch the requests session in HF Hub
        if hasattr(huggingface_hub, 'utils'):
            original_get_session = getattr(huggingface_hub.utils, 'get_session', None)
            if original_get_session:
                def patched_get_session():
                    session = original_get_session()
                    session.verify = False
                    session.timeout = 30
                    return session
                huggingface_hub.utils.get_session = patched_get_session
                print("✅ HuggingFace Hub patched")
        
        # Also try to patch the file_download module
        if hasattr(huggingface_hub, 'file_download'):
            original_get_session = getattr(huggingface_hub.file_download, 'get_session', None)
            if original_get_session:
                def patched_get_session():
                    session = original_get_session()
                    session.verify = False
                    session.timeout = 30
                    return session
                huggingface_hub.file_download.get_session = patched_get_session
                
    except ImportError:
        print("⚠️ HuggingFace Hub not available")
    
    # 7. Test connection
    print("\n🧪 Testing connection to HuggingFace...")
    try:
        import requests
        response = requests.get('https://huggingface.co/api/models?limit=1', verify=False, timeout=10)
        if response.status_code == 200:
            print("✅ Connection to HuggingFace successful!")
        else:
            print(f"⚠️ Connection returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
    
    print("\n🎉 SSL bypass setup complete!")
    print("Now restart your Jupyter kernel and run the notebook.")

if __name__ == "__main__":
    setup_aggressive_ssl_bypass()
