import torch
from CNN_feature import resnet50_cifar100_feature,FeatureExtractor,CNN_retrieve_topk
from traditional_feature import tradition_retrieve_topk
from feature_fusion import fusion_retrieve_topk

def main():
    # 参数设置
    seed = 2026
    metric = "cosine"
    num_query = 5
    phase = "color"

    # CNN深度特征检索
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50_cifar100_feature().to(device)
    ckpt = torch.load("/home/krli/IIP/checkpoints/resnet50_cifar100_layer4_fc.pth", map_location=device)
    model.load_state_dict(ckpt, strict=False)
    extractor = FeatureExtractor(model).to(device)
    CNN_retrieve_topk(extractor, num_query=num_query, topk=5, metric=metric, seed=seed)

    # 传统特征检索
    tradition_retrieve_topk(num_query=num_query, topk=5, DISTANCE=metric, phase=phase,seed=seed)

    # 融合特征检索
    fusion_retrieve_topk(num_query, topk=5, DISTANCE=metric, seed=seed)

if __name__ == "__main__":
    main()


