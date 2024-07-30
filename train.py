import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE, TransD, TransH
from openke.module.loss import MarginLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


embed_model = 'TransH'
percent = 0.45
tri_file = f'./benchmarks/FB15K237/remain_{percent}_unlearning.txt'
# dataloader for training
if embed_model == 'RotatE':
    train_dataloader = TrainDataLoader(
    	in_path=None,
    	tri_file=tri_file,
        # tri_file='./benchmarks/FB15K237/remaining_0.1.txt',
        # tri_file='./benchmarks/FB15K237/train2id.txt',
        ent_file="./benchmarks/FB15K237/entity2id.txt",
        rel_file="./benchmarks/FB15K237/relation2id.txt",
    	nbatches = 100,
    	threads = 8, 
        sampling_mode = "cross", 
    	bern_flag = 0, 
    	filter_flag = 1, 
    	neg_ent = 64,
    	neg_rel = 0)
    embeddings = RotatE(
    	ent_tot = train_dataloader.get_ent_tot(),
    	rel_tot = train_dataloader.get_rel_tot(),
    	dim = 200,
    	margin = 6.0,
    	epsilon = 2.0)
elif embed_model == 'TransD':
    train_dataloader = TrainDataLoader(
    	in_path=None,
    	tri_file=tri_file,
        # tri_file='./benchmarks/FB15K237/remaining_0.1.txt',
        # tri_file='./benchmarks/FB15K237/train2id.txt',
        ent_file="./benchmarks/FB15K237/entity2id.txt",
        rel_file="./benchmarks/FB15K237/relation2id.txt",
    	nbatches = 100,
    	threads = 8, 
    	sampling_mode = "normal", 
    	bern_flag = 1, 
    	filter_flag = 1, 
    	neg_ent = 25,
    	neg_rel = 0)
    embeddings = TransD(
    	ent_tot = train_dataloader.get_ent_tot(),
    	rel_tot = train_dataloader.get_rel_tot(),
    	dim_e = 200, 
    	dim_r = 200, 
    	p_norm = 1, 
    	norm_flag = True)
else:
    train_dataloader = TrainDataLoader(
    	in_path=None,
    	tri_file=tri_file,
        # tri_file='./benchmarks/FB15K237/remaining_0.1.txt',
        # tri_file='./benchmarks/FB15K237/train2id.txt',
        ent_file="./benchmarks/FB15K237/entity2id.txt",
        rel_file="./benchmarks/FB15K237/relation2id.txt",
    	nbatches = 100,
    	threads = 8, 
    	sampling_mode = "normal", 
    	bern_flag = 1, 
    	filter_flag = 1, 
    	neg_ent = 25,
    	neg_rel = 0)
    embeddings = TransH(
    	ent_tot = train_dataloader.get_ent_tot(),
    	rel_tot = train_dataloader.get_rel_tot(),
    	dim = 200, 
    	p_norm = 1, 
    	norm_flag = True)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")
embeddings.to('cuda:1')
# define the loss function
model = NegativeSampling(
	model = embeddings, 
    # loss = SigmoidLoss(adv_temperature = 2),
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)
# embeddings.load_checkpoint('./checkpoint/FB15K237/Retrain_Edges_RotatE.ckpt')
# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True
                  , opt_method = "adam")
# embeddings.load_checkpoint('./checkpoint/FB15K237/Retrain_FB15K237_Nodes_RotatE.ckpt')
trainer.run()
embeddings.save_checkpoint(f'./checkpoint/FB15K237/Retrain_FB15K237_{percent}_{embed_model}.ckpt')

# test the model
embeddings.load_checkpoint(f'./checkpoint/FB15K237/Retrain_FB15K237_{percent}_{embed_model}.ckpt')
tester = Tester(model = embeddings, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)