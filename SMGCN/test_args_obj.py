from models.pinsage_pytorch.graph_builder import HomoGraphBuilder


if __name__ == "__main__":
    import torch
    from models.pinsage_pytorch.pinsage_parser import parse_args
   
    args = parse_args()

    print(str(args))