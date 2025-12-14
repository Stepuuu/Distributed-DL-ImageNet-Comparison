import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 核心验证函数：必须放在顶层，方便多进程调用
def verify_one_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return None  # 返回 None 表示图片是好的
    except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
        return file_path  # 返回坏文件的路径

def check_images_parallel(root_dir, num_workers=32):
    print(f"正在扫描文件列表: {root_dir} ...")
    
    # 1. 先快速收集所有图片路径
    all_images = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.jpeg', '.jpg', '.png')):
                all_images.append(os.path.join(root, f))
    
    total_files = len(all_images)
    print(f"扫描完成，共找到 {total_files} 个图片。")
    print(f"启动 {num_workers} 个进程进行并行检查...")

    bad_files = []
    
    # 2. 使用多进程并行检查
    # chunksize 对性能有影响，对于大量小任务，设置稍微大一点可以减少进程间通信开销
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 使用 tqdm 显示进度
        results = list(tqdm(
            executor.map(verify_one_image, all_images, chunksize=100), 
            total=total_files, 
            unit="img",
            desc="并行检查中"
        ))

    # 3. 过滤出坏文件 (非 None 的结果)
    bad_files = [res for res in results if res is not None]

    print(f"\n检查完成！共发现 {len(bad_files)} 个损坏文件。")
    return bad_files

if __name__ == '__main__':
    # 你的服务器有 60+ 核，我们可以设置 32 或 48 个 worker
    # 注意：如果是机械硬盘，worker 太多可能会导致磁盘 IO 拥堵，建议 16-32 之间
    NUM_WORKERS = 32 

    print("=" * 60)
    print("开始多进程极速检查")
    print("=" * 60)

    # 检查 Train
    bad_list_train = check_images_parallel("./train", num_workers=NUM_WORKERS)
    
    # 检查 Val
    bad_list_val = check_images_parallel("./val", num_workers=NUM_WORKERS)

    # 合并结果
    all_bad_files = bad_list_train + bad_list_val

    # 删除逻辑
    if all_bad_files:
        print("\n" + "=" * 60)
        print(f"总计发现 {len(all_bad_files)} 个损坏文件:")
        for f in all_bad_files:
            print(f"  - {f}")
        
        confirm = input("\n是否立即删除这些文件？(y/n): ")
        if confirm.lower() == 'y':
            print("正在删除...")
            for f in all_bad_files:
                try:
                    os.remove(f)
                    print(f"已删除: {f}")
                except Exception as e:
                    print(f"删除失败 {f}: {e}")
            print("清理完成！")
        else:
            print("未删除文件。")
    else:
        print("\n完美！没有发现损坏的图片。")
