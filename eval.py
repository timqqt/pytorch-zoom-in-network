import torch
import numpy as np
from utils.utils import save_checkpoint, load_checkpoint, save_metrics, load_metrics, get_activation

def eval():
    sample_count = 0.0
    running_acc = 0.0
    running_loss = 0.0
    valid_running_acc = 0.0
    valid_running_loss = 0.0
    global_step = 0
    best_ACC = 0
    load_checkpoint(os.path.join(saved_path, 'con_histo_cross_'+str(i+1)+'_test1.pt'), model)
    model.eval()
    with torch.no_grad():                    

        # validation loop
        print("Begin Validation")
        valid_sample_count = 0.0
        valid_running_loss = 0.0
        valid_running_acc = 0.0
        for val_batch in tqdm.tqdm(valid_loader.generator()):
            val_input_reader = [e_reader for e_reader, _ in val_batch]
            labels = torch.stack([torch.Tensor([e_label]) for _, e_label in val_batch]).squeeze(1)
            labels = labels.type(torch.LongTensor).to(device)
            output, _ = model(val_input_reader)
            loss = criterion(output, labels)
            valid_running_loss += loss.item()
            _, pred_labels = output.data.cpu().topk(1, dim=1)
            valid_running_acc += torch.sum(pred_labels.t().squeeze() == labels.data.cpu().squeeze()).item()
            valid_sample_count += labels.shape[0]
    # evaluation
    average_valid_loss = valid_running_loss / len(valid_loader)
    average_valid_acc = valid_running_acc / valid_sample_count

    # print progress
    print('Valid Loss: {:.4f}, Valid Acc: {:.4f}'
          .format(average_valid_loss, average_valid_acc))
    print("Max memory used: {} Mb ".format(torch.cuda.memory_allocated(device=0)/ (1000 * 1000)))
