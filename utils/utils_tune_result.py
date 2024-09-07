import os
import json
import pandas as pd

def extract_data_from_json(file_path):
    all_data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                extracted_data = {  # 需要修改的参数
                    # 实验输出
                    'trial_id': data.get('trial_id'),
                    'eval_i2t_top1': data.get('eval_i2t_top1'),
                    'eval_i2t_top5': data.get('eval_i2t_top5'),
                    'eval_i2t_top10': data.get('eval_i2t_top10'),
                    'eval_t2i_top1': data.get('eval_t2i_top1'),
                    'eval_t2i_top5': data.get('eval_t2i_top5'),
                    'eval_t2i_top10': data.get('eval_t2i_top10'),
                    'eval_top1_avg': data.get('eval_top1_avg'),
                    'epoch': data.get('epoch'),
                    'objective': data.get('objective'),
                    # 超参数配置
                    'learning_rate': data.get('config', {}).get('learning_rate'),
                    'hidden_act': data.get('config', {}).get('hidden_act'),
                    'hidden_dropout_prob': data.get('config', {}).get('hidden_dropout_prob'),
                    'logit_scale': data.get('config', {}).get('logit_scale'),
                    'logit_bias': data.get('config', {}).get('logit_bias'),
                    'momentum': data.get('config', {}).get('momentum'),
                    'alpha': data.get('config', {}).get('alpha'),
                    'label_smoothing_factor': data.get('config', {}).get('label_smoothing_factor')
                }
                all_data.append(extracted_data)
            except json.JSONDecodeError:
                print(f"警告：无法解析JSON行：{line}")
    return all_data

def process_experiment_folders(root_dir):
    all_data = []
    
    for folder in os.listdir(root_dir):
        if folder.startswith('_objective'):
            json_path = os.path.join(root_dir, folder, 'result.json')
            if os.path.exists(json_path):
                data = extract_data_from_json(json_path)
                all_data.extend(data)
    
    return all_data

def main():
    root_directory = '~/ray_results/_objective_2024-09-07_00-14-28'  # ray 输出目录 # 需要修改的参数
    try:
        data = process_experiment_folders(root_directory)
        
        if not data:
            print("警告：没有找到任何数据。请检查指定的目录是否正确。")
            return

        # 创建 DataFrame
        df = pd.DataFrame(data)
        
        # 导出到 Excel
        output_file = '~/ray_results/experiment_results_e1.xlsx' # 输出文件名 # 需要修改的参数
        df.to_excel(output_file, index=False)
        print(f"数据已成功导出到 {output_file}")
        print(f"总共处理了 {len(data)} 条数据记录。")
    except Exception as e:
        print(f"处理数据时发生错误：{str(e)}")

if __name__ == "__main__":
    """
    检索 "需要修改的参数" 按需修改
    python utils/utils_tune_result.py
    """
    main()