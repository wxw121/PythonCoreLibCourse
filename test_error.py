try:
    import scikit_learn_tutorial.sklearn_advanced as sa
    sa.ensemble_learning_example()
except Exception as e:
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {str(e)}")
    import traceback
    traceback.print_exc()
