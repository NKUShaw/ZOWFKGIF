import openke
import time
import torch
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransD, TransH, RotatE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader



def calculate_gradients(model, data):
    model.eval()

    loss = model.model({
        'batch_h': torch.autograd.Variable(torch.from_numpy(data['batch_h']).cuda()),
        'batch_t': torch.autograd.Variable(torch.from_numpy(data['batch_t']).cuda()),
        'batch_r': torch.autograd.Variable(torch.from_numpy(data['batch_r']).cuda()),
        'batch_y': torch.autograd.Variable(torch.from_numpy(data['batch_y']).cuda()),
        'mode': data['mode']
    })
    total_memory = 0
    for i in range(torch.cuda.device_count()):
        total_memory += torch.cuda.max_memory_allocated(i) / (1024 ** 2)
    print(f"Total LOSS memory usage across all GPUs: {total_memory} MB")
    loss_scalar = torch.mean(loss)
    params_to_update = [param for name, param in model.named_parameters() if name.endswith('.weight')]
    grads = torch.autograd.grad(loss_scalar, params_to_update, create_graph=True)
    del loss, loss_scalar
    torch.cuda.empty_cache()
    return grads


def calculate_zo_gradients(model, data, epsilon=1e-5):
    model.eval()
    grads = []
    params_to_update = [param for name, param in model.named_parameters() if
                        name.endswith('.weight') and param.requires_grad]
    for param in params_to_update:
        grad = torch.zeros_like(param.data).cuda()
        perturb = torch.randn(param.data.shape[1]).cuda() * epsilon
        perturb = perturb.unsqueeze(0).expand(param.data.shape)
        original_param = param.data.clone()

        param.data.add_(perturb)
        with torch.no_grad():
            loss_pos = model.model({
                'batch_h': torch.from_numpy(data['batch_h']).to('cuda'),
                'batch_t': torch.from_numpy(data['batch_t']).to('cuda'),
                'batch_r': torch.from_numpy(data['batch_r']).to('cuda'),
                'batch_y': torch.from_numpy(data['batch_y']).to('cuda'),
                'mode': data['mode']
            })
        loss_pos_scalar = torch.mean(loss_pos)

        param.data.copy_(original_param)
        param.data.sub_(perturb)
        with torch.no_grad():
            loss_neg = model.model({
                'batch_h': torch.from_numpy(data['batch_h']).to('cuda'),
                'batch_t': torch.from_numpy(data['batch_t']).to('cuda'),
                'batch_r': torch.from_numpy(data['batch_r']).to('cuda'),
                'batch_y': torch.from_numpy(data['batch_y']).to('cuda'),
                'mode': data['mode']
            })
        loss_neg_scalar = torch.mean(loss_neg)
    
        param.data.copy_(original_param)

        grad = (loss_pos_scalar - loss_neg_scalar) / (2 * epsilon) * perturb
        grads.append(grad.mean(dim=0))

        torch.cuda.empty_cache()
    return grads

def hvps(grad_all, model_params, h_estimate):
    element_product = 0
    for grad_elem, v_elem in zip(grad_all, h_estimate):
        element_product += torch.sum(grad_elem * v_elem)
    return_grads = torch.autograd.grad(element_product, model_params, create_graph=True)
    del element_product
    torch.cuda.empty_cache()
    return return_grads

def woodfisher_hvps(grad_all, gamma=1.0):
    hessian_product = tuple()
    vTv = sum(torch.sum(grad_elem * grad_elem) for grad_elem in grad_all)
    for grad_elem in grad_all:
        term1 = (1 / gamma) * grad_elem
        term2 = (gamma ** (-2) * grad_elem * vTv) / (1 + gamma ** (-1) * vTv)
        hessian_estimate = term1 + term2
        hessian_product += (hessian_estimate,)
    return hessian_product


def update_and_save_checkpoint(checkpoint_path, new_checkpoint_path, new_params):
    weights = torch.load(checkpoint_path)
    weights['ent_embeddings.weight'] = new_params[0]
    weights['rel_embeddings.weight'] = new_params[1]
    torch.save(weights, new_checkpoint_path)
    print(f"Updated checkpoint saved to {new_checkpoint_path}")

def GIF_unleanring(model, train_dataloader, test_dataloader, epsilon=None, gamma= None, iteration=100, damp=0.0, scale=50):
    start_time = time.time()

    for data in train_dataloader:
        grad_full = calculate_gradients(model, data)
        # grad_full = calculate_zo_gradients(model, data, epsilon=epsilon)
        break
    total_memory = 0
    for i in range(torch.cuda.device_count()):
        total_memory += torch.cuda.max_memory_allocated(i) / (1024 ** 2)
    print(f"Total GradsFull memory usage across all GPUs: {total_memory} MB")
    for data in test_dataloader:
        grad_removed = calculate_gradients(model, data)
        # grad_removed = calculate_zo_gradients(model, data, epsilon=epsilon)
        break
    total_memory = 0
    for i in range(torch.cuda.device_count()):
        total_memory += torch.cuda.max_memory_allocated(i) / (1024 ** 2)
    print(f"Total GradsRemoved memory usage across all GPUs: {total_memory} MB")
    grad1 = [g1 - g2 for g1, g2 in zip(grad_full, grad_removed)]
    grad2 = grad_removed
    res_tuple = (grad_full, grad1, grad2)

    v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
    h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
    for i in range(iteration):
        model_params = [p for p in model.parameters() if p.requires_grad]
        hv = woodfisher_hvps(res_tuple[0], gamma=gamma)
        # hv = hvps(res_tuple[0], model_params, h_estimate)
        with torch.no_grad():
            h_estimate = [v1 + (1 - damp) * h_estimate1 - hv1 / scale for v1, h_estimate1, hv1 in
                          zip(v, h_estimate, hv)]

    print(f"final h_estimate: {torch.cuda.max_memory_allocated() / (1024 ** 2)} MB")
    params_change = [h_est / scale for h_est in h_estimate]
    params_esti = [p1 + p2 for p1, p2 in zip(params_change, model_params)]
    total_memory = 0
    for i in range(torch.cuda.device_count()):
        total_memory += torch.cuda.max_memory_allocated(i) / (1024 ** 2)
    print(f"Total memory usage across all GPUs: {total_memory} MB")
    del grad_full, grad_removed, res_tuple, v, h_estimate, params_change
    torch.cuda.empty_cache()

    print(time.time() - start_time)

    return params_esti

train_dataloader = TrainDataLoader(
    in_path = None,
    tri_file = "./benchmarks/YAGO3-10/train2id.txt",
    ent_file = "./benchmarks/YAGO3-10/entity2id.txt",
    rel_file = "./benchmarks/YAGO3-10/relation2id.txt",
    nbatches = 100,
    threads = 8,
    sampling_mode = "normal",
    bern_flag = 1,
    filter_flag = 1,
    neg_ent = 25,
    neg_rel = 0)
print('----------------------------------------------------------------------')
retrain_dataloader = TrainDataLoader(
    in_path = None,
    tri_file = './benchmarks/YAGO3-10/remaining_0.1.txt',
    ent_file = "./benchmarks/YAGO3-10/entity2id.txt",
    rel_file = "./benchmarks/YAGO3-10/relation2id.txt",
    nbatches = 100,
    threads = 8,
    sampling_mode = "normal",
    bern_flag = 1,
    filter_flag = 1,
    neg_ent = 25,
    neg_rel = 0)

# model = TransE(
#   ent_tot = train_dataloader.get_ent_tot(),
#   rel_tot = train_dataloader.get_rel_tot(),
#   dim = 200,
#   p_norm = 1,
#   norm_flag = True)

# model = TransD(
#   ent_tot = train_dataloader.get_ent_tot(),
#   rel_tot = train_dataloader.get_rel_tot(),
#   dim_e = 200,
#   dim_r = 200,
#   p_norm = 1,
#   norm_flag = True)

# model = TransH(
#     ent_tot = train_dataloader.get_ent_tot(),
#     rel_tot = train_dataloader.get_rel_tot(),
#     dim = 200,
#     p_norm = 1,
#     norm_flag = True)

model = RotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200,
	margin = 6.0,
	epsilon = 2.0,
)
checkpoint_path="./checkpoint/YAGO3-10/YAGO3-10_RotatE.ckpt"
model = torch.nn.DataParallel(model)
model.to('cuda')
model.module.load_checkpoint(checkpoint_path)
model = NegativeSampling(
    model = model,
    loss = MarginLoss(margin = 5.0),
    batch_size = train_dataloader.get_batch_size()
)

epsilon=1e-5
gamma=1.00
iteration=100
damp=0.00
scale=500
print('epsilon:', epsilon)
print('gamma:', gamma)
print('iteration:', iteration)
print('damp:', damp)
print('scale:', scale)
# new_checkpoint_path=f"./checkpoint/YAGO3-10/GIF/ZOWFGIF_{epsilon}_RotatE.ckpt"
new_checkpoint_path = f"./checkpoint/YAGO3-10/GNNDelete_Random_RotatE_YAGO3-10.ckpt"
params_esti = GIF_unleanring(model, train_dataloader, retrain_dataloader, epsilon=epsilon, gamma=gamma, iteration=iteration, damp=damp, scale=scale)

update_and_save_checkpoint(checkpoint_path=checkpoint_path,
                           new_checkpoint_path=new_checkpoint_path,
                           new_params=params_esti)

test_dataloader = TestDataLoader("./benchmarks/YAGO3-10/", "link")
# define the model
# unlearn_model = TransE(
#   ent_tot = train_dataloader.get_ent_tot(),
#   rel_tot = train_dataloader.get_rel_tot(),
#   dim = 200,
#   p_norm = 1,
#   norm_flag = True)
# unlearn_model = TransD(
#   ent_tot = train_dataloader.get_ent_tot(),
#   rel_tot = train_dataloader.get_rel_tot(),
#   dim_e = 200,
#   dim_r = 200,
#   p_norm = 1,
#   norm_flag = True)
# unlearn_model = TransH(
#     ent_tot = train_dataloader.get_ent_tot(),
#     rel_tot = train_dataloader.get_rel_tot(),
#     dim = 200,
#     p_norm = 1,
#     norm_flag = True)
unlearn_model = RotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200,
	margin = 6.0,
	epsilon = 2.0,
)
# unlearn_transe = torch.nn.DataParallel(unlearn_transe)
# test the model
unlearn_model.load_checkpoint(new_checkpoint_path)
unlearn_tester = Tester(model = unlearn_model, data_loader = test_dataloader, use_gpu = True)
unlearn_tester.run_link_prediction(type_constrain = False)
