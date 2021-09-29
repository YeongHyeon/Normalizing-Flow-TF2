import os
import numpy as np
import source.utils as utils

def make_result(agent, dataset, epoch, savedir):

    utils.make_dir(path=os.path.join(savedir, 'reconstruction'), refresh=False)
    utils.make_dir(path=os.path.join(savedir, 'energy'), refresh=False)

    minibatch = dataset.next_batch(batch_size=100, ttv=2)
    dataset.idx_val = 0
    step_dict = agent.step(minibatch=minibatch, training=False)

    dict_plot = {}
    for idx_batch in range(10):
        dict_plot[idx_batch] = \
            {'y':minibatch['x'][idx_batch][:, :, 0], 'y_hat':step_dict['y_hat'][idx_batch][:, :, 0]}
    utils.plot_generation(dict_plot=dict_plot, savepath=os.path.join(savedir, 'reconstruction', "epoch_%08d.png" %(epoch)))
    utils.plot_scatter(z_0=step_dict['z_0'], z_k=step_dict['z_k'], savepath=os.path.join(savedir, 'energy', "epoch_%08d.png" %(epoch)))

def training(agent, dataset, batch_size, epochs):

    savedir = 'results_tr'
    utils.make_dir(path=savedir, refresh=True)

    print("\n** Training to %d epoch | Batch size: %d" %(epochs, batch_size))
    iteration = 0
    loss_best = 1e+12

    make_result(agent=agent, dataset=dataset, epoch=0, savedir=savedir)
    for epoch in range(epochs):

        list_loss = []
        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, ttv=0)
            if(minibatch['x'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, iteration=iteration, training=True)
            list_loss.append(step_dict['losses']['loss_mean'])
            iteration += 1
            if(minibatch['terminate']): break
        list_loss = np.asarray(list_loss)
        loss_tmp = np.average(list_loss)

        print("Epoch [%d / %d] | Loss: %f" %(epoch, epochs, loss_tmp))

        make_result(agent=agent, dataset=dataset, epoch=epoch+1, savedir=savedir)

        if(loss_best > loss_tmp):
            loss_best = loss_tmp
            agent.save_params(model='model_1_best_loss')
        agent.save_params(model='model_0_finepocch')

def test(agent, dataset):

    savedir = 'results_te'
    utils.make_dir(path=savedir, refresh=True)

    list_model = utils.sorted_list(os.path.join('Checkpoint', 'model*'))
    for idx_model, path_model in enumerate(list_model):
        list_model[idx_model] = path_model.split('/')[-1]

    loss_best = 1e+12
    for idx_model, path_model in enumerate(list_model):

        print("\n** Test with %s" %(path_model))
        agent.load_params(model=path_model)
        utils.make_dir(path=os.path.join(savedir, path_model), refresh=False)

        list_loss = []
        idx_save = 0
        while(True):

            minibatch = dataset.next_batch(batch_size=1, ttv=1)
            if(minibatch['x'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, training=False)

            for idx_batch in range(minibatch['y'].shape[0]):
                savename = "%d_%08d.png" %(np.argmax(minibatch['y'][idx_batch]), idx_save)
                utils.plot_comparison(y=minibatch['x'][idx_batch][:, :, 0], y_hat=step_dict['y_hat'][idx_batch][:, :, 0], \
                    savepath=os.path.join(savedir, path_model, savename))
                idx_save += 1

            list_loss.append(step_dict['losses']['loss_mean'])
            if(minibatch['terminate']): break
        list_loss = np.asarray(list_loss)
        loss_tmp = np.average(list_loss)
