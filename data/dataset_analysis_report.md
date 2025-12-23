# 数据集分析报告

## 文件分析

根据 `/comp_robot/zhoujiazhou/projects/Active-Coconut/data/viscot_363k_lvr_formatted.json` 和 `/comp_robot/zhoujiazhou/projects/Active-Coconut/data/viscot_sroie_dude_lvr_formatted.json` 的分析：

## 数据集统计

### viscot_363k_lvr_formatted.json
根据 `dataset_paths_summary.md` 和 `extract_dataset_paths.py` 的分析，该文件包含 **10个数据集**：

1. **cub** - 3,987条记录
2. **docvqa** - 33,453条记录
3. **flickr30k** - 135,735条记录
4. **gqa** - 88,294条记录
5. **infographicsvqa** - 15,055条记录
6. **openimages** - 43,053条记录
7. **textcap** - 32,152条记录
8. **textvqa** - 18,524条记录
9. **v7w** - 30,491条记录
10. **vsr** - 3,376条记录

### viscot_sroie_dude_lvr_formatted.json
根据文件名推断，该文件应包含：
- **sroie** 数据集
- **dude** 数据集

（具体记录数需要进一步分析，但文件大小约为5.6MB，而viscot_363k约为148MB）

## 数据集相对路径映射

所有数据集的相对路径都相对于：`/comp_robot/zhoujiazhou/Datasets/Visual_cot`

| 数据集名称 | 相对路径 | 说明 |
|-----------|---------|------|
| cub | `images/cub/CUB_200_2011/images` | 包含子目录结构 |
| docvqa | `images/docvqa` | |
| flickr30k | `images/flickr30k/flickr30k-images` | |
| gqa | `images/gqa/images` | |
| infographicsvqa | `images/infographicsvqa` | |
| openimages | `images/openimages` | |
| textcap | `images/textvqa/train_images` | 与textvqa共享目录 |
| textvqa | `images/textvqa/train_images` | 与textcap共享目录 |
| v7w | `images/visual7w/images` | |
| vsr | `images/vsr/images` | |
| sroie | `images/viscot/sroie` | 推测路径 |
| dude | `images/viscot/dude` | 推测路径 |

## JSON文件中的Key结构

根据 `extract_dataset_paths.py` 代码分析，JSON文件中的数据结构为：
- 每个数据项是一个字典
- 包含 `dataset` key（数据集名称）
- 包含 `image` key（图像路径，可能是字符串或列表）
- 图像路径格式：`viscot/{dataset_name}/{path_to_file}`

## 总结

- **viscot_363k_lvr_formatted.json**: 包含10个数据集，共约404,120条记录
- **viscot_sroie_dude_lvr_formatted.json**: 至少包含2个数据集（sroie和dude）
- **总共涉及的数据集**: 至少12个不同的数据集
