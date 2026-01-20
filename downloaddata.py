import kagglehub

# download dataset
path = kagglehub.dataset_download("nudratabbas/cryptocurrency-market-snapshot-top-250-coins")

print("Path to dataset files:", path)