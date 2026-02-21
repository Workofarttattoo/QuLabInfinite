#!/usr/bin/env python3
"""
Simple test to verify ModelScope installation and functionality
"""

def test_modelscope_import():
    """Test ModelScope import"""
    try:
        import modelscope
        print(f"✅ ModelScope imported successfully! Version: {modelscope.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import ModelScope: {e}")
        return False

def test_modelscope_components():
    """Test specific ModelScope components"""
    try:
        from modelscope import AutoTokenizer, AutoModelForCausalLM
        print("✅ ModelScope AutoTokenizer and AutoModelForCausalLM imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Failed to import ModelScope components: {e}")
        return False

def test_transformers_fallback():
    """Test transformers as fallback"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        print("✅ Transformers imported successfully!")
        print(f"✅ PyTorch imported! CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import transformers: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing ModelScope and dependencies...")
    print("-" * 50)
    
    modelscope_ok = test_modelscope_import()
    components_ok = test_modelscope_components() if modelscope_ok else False
    transformers_ok = test_transformers_fallback()
    
    print("\n📊 Summary:")
    print(f"ModelScope: {'✅' if modelscope_ok else '❌'}")
    print(f"ModelScope Components: {'✅' if components_ok else '❌'}")
    print(f"Transformers Fallback: {'✅' if transformers_ok else '❌'}")
    
    if modelscope_ok and components_ok:
        print("\n🎉 Ready for Qwen model loading!")
    elif transformers_ok:
        print("\n⚠️ Can use Qwen via transformers as fallback")
    else:
        print("\n❌ Dependencies not properly installed")