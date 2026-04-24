import subprocess
import os
import pickle
import numpy as np

def run_search():
    lrs = [0.1, 0.01, 0.001]
    hidden_dims = [128, 256, 512]
    weight_decays = [1e-4, 1e-3]
    
    results = []
    
    print(f"{'LR':<8} | {'Hidden':<8} | {'WD':<8} | {'Best Val Acc':<12}")
    print("-" * 45)
    
    for lr in lrs:
        for hidden_dim in hidden_dims:
            for wd in weight_decays:
                save_path = f"model_lr{lr}_h{hidden_dim}_wd{wd}.pkl"
                cmd = [
                    "python3", "train.py",
                    "--lr", str(lr),
                    "--hidden_dim", str(hidden_dim),
                    "--weight_decay", str(wd),
                    "--epochs", "5",  # 搜索时跑 5 轮即可看出趋势
                    "--save_path", save_path
                ]
                
                # 运行训练
                subprocess.run(cmd, stdout=subprocess.DEVNULL) # 隐藏训练细节输出，保持搜索界面整洁
                
                # 读取结果
                try:
                    with open(save_path, 'rb') as f:
                        data = pickle.load(f)
                        best_val_acc = max(data['history']['val_acc'])
                        results.append({
                            'lr': lr,
                            'hidden': hidden_dim,
                            'wd': wd,
                            'val_acc': best_val_acc
                        })
                        print(f"{lr:<8} | {hidden_dim:<8} | {wd:<8} | {best_val_acc:.4f}")
                except Exception as e:
                    print(f"{lr:<8} | {hidden_dim:<8} | {wd:<8} | FAILED (likely NaN)")

    # 打印最终排行榜
    print("\n" + "="*20 + " RANKING " + "="*20)
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    for i, res in enumerate(results[:5]):
        print(f"Top {i+1}: LR={res['lr']}, Hidden={res['hidden']}, WD={res['wd']} -> Acc: {res['val_acc']:.4f}")

if __name__ == '__main__':
    run_search()
