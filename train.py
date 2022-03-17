def train(model, train_loader, valid_loader):
    model_dir = '/hpc/group/carin/fk43/FanjieKong/Megapixels/PytorchATS/new_histo_test_1/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    #criterion = nn.BCELoss ()
    criterion = nn.CrossEntropyLoss()
    num_epochs = 100
    eval_every_epochs = 1
    eval_every = len(TrainingDataReader) * eval_every_epochs
    file_path = model_dir
    saved_path = model_dir
    best_valid_loss = float("Inf")
    training_show_every = len(TrainingDataReader) * 0.1
    clip = 5.0
    sample_count = 0.0
    running_acc = 0.0
    running_loss = 0.0
    valid_running_acc = 0.0
    valid_running_loss = 0.0
    global_step = 0
    best_ACC = 0
    train_acc_list = []
    valid_acc_list = []
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    
    contrastive_learning = True
    apply_con_epochs = 10
    
    for epoch in range(num_epochs):
        sample_count = 0.0
        running_acc = 0.0
        running_loss = 0.0
        model.train()
        for train_batch in tqdm.tqdm(train_loader.generator()):
            train_input_reader = [e_reader for e_reader, _ in train_batch]
            labels = torch.stack([torch.Tensor([e_label]) for _, e_label in train_batch]).squeeze(1)
            labels = labels.type(torch.LongTensor).to(device)
            if contrastive_learning and epoch >= apply_con_epochs and torch.sum(labels) > 0:

                pos_reader = np.array(train_input_reader)[labels.data.cpu().numpy() == 1]
                pos_output, pos_sparse_loss = model(pos_reader)

                con_output, con_sparse_loss = model._compute_constrastive_predictions(pos_reader)
                
                pos_loss = criterion(pos_output, labels[labels == 1]) + pos_sparse_loss
                con_loss = criterion(con_output, torch.zeros_like(labels[labels == 1]).to(device)) + con_sparse_loss
                
                pos_weight = len(pos_reader)/len(train_input_reader)
                if torch.sum(1-labels) == 0:
                    neg_loss = 0 
                    neg_weight = 0
                    neg_output = None
                    output = pos_output
                    labels = labels[labels == 1]
                else: 
                    neg_reader = np.array(train_input_reader)[labels.data.cpu().numpy() == 0]
                    neg_output, neg_sparse_loss = model(neg_reader)
                    neg_loss = criterion(neg_output, labels[labels == 0]) + neg_sparse_loss
                    neg_weight = len(neg_reader)/(len(neg_reader) + len(pos_reader))
                    output = torch.cat([pos_output, neg_output], dim=0)
                    labels =  torch.cat([labels[labels == 1], labels[labels == 0]], dim=0)
                    
                loss = pos_weight * pos_loss + (1-pos_weight)* (neg_weight * neg_loss + (1-neg_weight) *con_loss)

            else:
                output, sparse_loss = model(train_input_reader)
                loss = criterion(output, labels) + sparse_loss
            
            #loss = criterion(sigfunc(output), labels) + sparse_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()

            # update running values
            #pred_labels = np.where(output.data.cpu() > 0.5, 1, 0)
            _, pred_labels = output.data.cpu().topk(1, dim=1)

            running_acc += torch.sum(pred_labels.t().squeeze() == labels.data.cpu().squeeze()).item()
            sample_count += labels.shape[0]
            running_loss += loss.item()
            global_step += 1

            # training stats
            if global_step % training_show_every == 0:
                average_running_acc = running_acc / sample_count
                average_train_loss =  running_loss / sample_count
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_running_acc))
            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # validation loop
                    print("Begin Validation")
                    valid_sample_count = 0.0
                    valid_running_loss = 0.0
                    valid_running_acc = 0.0
                    used_time = 0.0
                    for val_batch in tqdm.tqdm(valid_loader.generator()):
                        val_input_reader = [e_reader for e_reader, _ in val_batch]
                        labels = torch.stack([torch.Tensor([e_label]) for _, e_label in val_batch]).squeeze(1)
                        labels = labels.type(torch.LongTensor).to(device)
                        start_time = time.time()
                        output, _ = model(val_input_reader)
                        used_time += time.time() - start_time
                        loss = criterion(output, labels)

                        valid_running_loss += loss.item()
                        #pred_labels = np.where(output.data.cpu() > 0.5, 1, 0)
                        _, pred_labels = output.data.cpu().topk(1, dim=1)
                        valid_running_acc += torch.sum(pred_labels.t().squeeze() == labels.data.cpu().squeeze()).item()
                        valid_sample_count += labels.shape[0]
                # evaluation

                average_train_loss = running_loss / eval_every
                average_train_acc = running_acc / sample_count
                average_valid_loss = valid_running_loss / len(valid_loader)
                average_valid_acc = valid_running_acc / valid_sample_count
                train_loss_list.append(average_train_loss)
                train_acc_list.append(average_train_acc)
                valid_loss_list.append(average_valid_loss)
                valid_acc_list.append(average_valid_acc)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()
                # print progress
                print('Iter [{}/{}], Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.4f}'
                      .format(i+1, total_cross,epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_train_acc, average_valid_loss, average_valid_acc))
                print("Max memory used: {} Mb ".format(torch.cuda.memory_allocated(device=0)/ (1024 * 1024)))
                print("Average Time per sample {} sec".format(used_time/(valid_sample_count)))
                # checkpoint
                if best_ACC <= average_valid_acc:
                    best_ACC = average_valid_acc
                    save_checkpoint(os.path.join(saved_path, 'con_histo_cross_'+str(i+1)+'_test1.pt'), model, best_valid_loss)
                    save_metrics(os.path.join(saved_path, 'con_histo_cross_'+str(i+1)+'_test1_metrics.pt'), train_loss_list, valid_loss_list, global_steps_list)
            # reinit
        scheduler.step()
    save_metrics(os.path.join(saved_path, 'con_histo_cross_'+str(i+1)+'_test1_metrics.pt'), train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training Cross ', str(i+1))
    ten_cross_acc_list.append(best_ACC)

print("Final 10 Cross Validation Results: ")
print(ten_cross_acc_list)
print(np.mean(ten_cross_acc_list))