import random  # 导入随机数模块
import math    # 导入数学计算模块
import numpy as np  # 导入numpy库用于数组操作


class MaskingGenerator:
    """生成随机掩码的工具类"""
    def __init__(
        self,
        input_size,          # 输入图像尺寸（高/宽）
        num_masking_patches=None,  # 总掩码块数
        min_num_patches=4,   # 最小单次掩码块数
        max_num_patches=None,  # 最大单次掩码块数
        min_aspect=0.3,      # 最小长宽比
        max_aspect=None,     # 最大长宽比
    ):
        # 处理输入尺寸格式
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width  # 总块数
        self.num_masking_patches = num_masking_patches  # 目标掩码块数

        # 设置最小/最大掩码块数
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        # 处理长宽比范围
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))  # 存储对数范围

    def __repr__(self):
        # 返回类实例的字符串表示
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def  get_shape(self):
        # 获取当前形状
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        # 实际生成掩码的内部方法
        delta = 0
        for _ in range(10):  # 尝试最多10次防止无限循环
            # 计算目标区域参数
            target_area = random.uniform(self.min_num_patches, max_mask_patches)  # 随机生成目标区域大小
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))  # 通过指数变换获得平滑的宽高比
            h = int(round(math.sqrt(target_area * aspect_ratio)))  # 计算高度（考虑宽高比）
            w = int(round(math.sqrt(target_area / aspect_ratio)))  # 计算宽度（反向调整）
            
            # 检查尺寸有效性（必须小于原始尺寸）
            if w < self.width and h < self.height:
                # 随机定位掩码区域
                top = random.randint(0, self.height - h)  # 确保不会越界
                left = random.randint(0, self.width - w)

                # 计算当前区域已掩码数量
                num_masked = mask[top : top + h, left : left + w].sum()
                # 处理新旧掩码区域的重叠部分
                if 0 < h * w - num_masked <= max_mask_patches:  # 确保有效掩码数量
                    # 更新掩码区域并统计新增掩码数
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1  # 标记为已掩码
                                delta += 1

                if delta > 0:
                    break  # 成功生成掩码后立即退出循环
        return delta  # 返回新增的掩码数量

    def __call__(self, num_masking_patches=0):
        # 主调用方法生成掩码
        mask = np.zeros(shape=self.get_shape(), dtype=bool)  # 初始化掩码数组
        mask_count = 0
        while mask_count < num_masking_patches:  # 循环直到满足数量要求
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)  # 限制最大值

            delta = self._mask(mask, max_mask_patches)  # 生成掩码
            if delta == 0:  # 无法继续生成时退出
                break
            else:
                mask_count += delta  # 更新计数

        return mask  # 返回生成的掩码