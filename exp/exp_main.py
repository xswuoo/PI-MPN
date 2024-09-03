from data_provider.data_factory import *
from exp.exp_basic import Exp_Basic
from models import PIMPN
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, MAE
from torch import optim
import torch
import random
import os
import time
import numpy as np

torch.autograd.set_detect_anomaly(True)

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.clip = 1

    def _build_model(self):
        model_dict = {
            'PIMPN': PIMPN,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        print(model)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_pickle(self):
        pickle_file = os.path.join(self.args.root_path, self.args.adj_path)
        pickle_data = load_pickle(pickle_file)
        predefined_A = torch.tensor(pickle_data[2]) - torch.eye(self.args.num_nodes)
        lat_lon = torch.tensor(pickle_data[3])
        return predefined_A, lat_lon

    def _get_poi(self):
        local_poi = load_poi(os.path.join(self.args.root_path, self.args.poi_path))
        local_poi = local_poi.transpose()
        local_poi = local_poi.drop(['Big Category', 'Mid Category', 'Count'])
        local_poi = torch.tensor(local_poi.values.astype(np.float32)).to(self.device)
        return local_poi

    def _io2od_flow(self, od_flow_mask, index):
        batch_size, pred_len = od_flow_mask.shape[:2]
        num_nodes = self.args.num_nodes
        index = index.expand(batch_size, pred_len, -1).to(self.device)
        od_flow_zeros = torch.zeros(batch_size, pred_len, num_nodes * num_nodes).to(self.device)
        od_flow_zeros.scatter_(2, index, od_flow_mask)
        od_flow = od_flow_zeros.view(batch_size, pred_len, 1, num_nodes, num_nodes)
        inflow = od_flow.sum(dim=4)
        outflow = od_flow.sum(dim=3)
        return inflow, outflow

    def _construct_field(self, lat_lon, field=9):
        lat_max, lat_min = lat_lon[:, 0].max(), lat_lon[:, 0].min()
        lon_max, lon_min = lat_lon[:, 1].max(), lat_lon[:, 1].min()
        lat_step = (lat_max - lat_min) / (field - 1)
        lon_step = (lon_max - lon_min) / (field - 1)
        lat_indices = torch.floor((lat_lon[:, 0] - lat_min) / lat_step).long()
        lon_indices = torch.floor((lat_lon[:, 1] - lon_min) / lon_step).long()
        indices = lon_indices * field + lat_indices
        return indices

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                 weight_decay=self.args.weight_decay)
        return model_optim

    def vali(self, vali_data, vali_loader, scaler, predefined_A, local_poi, lat_lon_index):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, columns_to_observe) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                local_stamp = batch_x_mark[:, :, -1]
                indices = lat_lon_index.expand(batch_x.size(0), self.args.pred_len, -1)
                batch_x_mark = batch_x_mark.unsqueeze(2).expand(-1, -1, self.args.num_pairs, -1)
                inputs = torch.cat((batch_x.unsqueeze(-1), batch_x_mark), dim=-1)
                inputs = inputs.permute(0, 3, 2, 1)

                outputs, outputs_io = self.model(inputs, predefined_A, local_poi, local_stamp, columns_to_observe,
                                                 indices)

                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = scaler.inverse_transform(outputs)
                batch_y = scaler.inverse_transform(batch_y)
                loss = MAE(outputs, batch_y)


                inflow_y, outflow_y = self._io2od_flow(batch_y, columns_to_observe)
                y_io = torch.cat((inflow_y, outflow_y), dim=2)
                loss_IO = MAE(outputs_io, y_io)
                loss = loss + self.args.lambda_ * loss_IO

                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        predefined_A, lat_lon = self._get_pickle()
        local_poi = self._get_poi() if self.args.use_poi else None
        lat_lon_index = self._construct_field(lat_lon).to(self.device)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        task_level = 1
        train_steps = len(train_loader)
        report_gap = int((train_data.data_x.shape[0] / self.args.batch_size / 5) // 1)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, columns_to_observe) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                local_stamp = batch_x_mark[:, :, -1]
                indices = lat_lon_index.expand(batch_x.size(0), self.args.pred_len, -1)
                batch_x_mark = batch_x_mark.unsqueeze(2).expand(-1, -1, self.args.num_pairs, -1)
                inputs = torch.cat((batch_x.unsqueeze(-1), batch_x_mark), dim=-1)
                inputs = inputs.permute(0, 3, 2, 1)

                outputs, outputs_io = self.model(inputs, predefined_A, local_poi, local_stamp, columns_to_observe,
                                                 indices)
                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                scaler = train_data.scaler
                outputs_loss = outputs
                batch_y_loss = batch_y

                if self.args.cl and (self.args.itr % self.args.cl_steps == 0) and task_level <= self.args.pred_len:
                    print("finish task_level: ", task_level)
                    task_level += 1

                if self.args.cl:
                    loss = MAE(outputs_loss[:, :task_level, :], batch_y_loss[:, :task_level, :])
                else:
                    loss = MAE(outputs_loss, batch_y_loss)

                inflow_x, outflow_x = self._io2od_flow(batch_x, columns_to_observe)
                inflow_y, outflow_y = self._io2od_flow(batch_y_loss, columns_to_observe)
                y_io = torch.cat((inflow_y, outflow_y), dim=2)
                loss_io = MAE(outputs_io, y_io)
                loss = loss + self.args.lambda_ * loss_io
                train_loss.append(loss.item())

                if (i + 2) % report_gap == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                model_optim.step()

                self.args.itr += 1
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, scaler, predefined_A, local_poi, lat_lon_index)
            test_loss = self.vali(test_data, test_loader, scaler, predefined_A, local_poi, lat_lon_index)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            f = open("log.txt", 'a')
            f.write("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            f.write('\n')
            f.close()
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(best_model_path, map_location=device))

        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        predefined_A, lat_lon = self._get_pickle()
        if self.args.use_poi:
            local_poi = self._get_poi()
        else:
            local_poi = None
        lat_lon_index = self._construct_field(lat_lon).to(self.device)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        preds_o = []
        trues_o = []
        preds_d = []
        trues_d = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        scaler = test_data.scaler

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, columns_to_observe) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                local_stamp = batch_x_mark[:, :, -1]
                indices = lat_lon_index.expand(batch_x.size(0), self.args.pred_len, -1)
                batch_x_mark = batch_x_mark.unsqueeze(2).expand(-1, -1, self.args.num_pairs, -1)
                inputs = torch.cat((batch_x.unsqueeze(-1), batch_x_mark), dim=-1)
                inputs = inputs.permute(0, 3, 2, 1)

                outputs, outputs_io = self.model(inputs, predefined_A, local_poi, local_stamp, columns_to_observe,
                                                 indices)
                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = scaler.inverse_transform(outputs)
                batch_y = scaler.inverse_transform(batch_y)

                outputs_o, outputs_d = self._io2od_flow(outputs, columns_to_observe)
                batch_y_o, batch_y_d = self._io2od_flow(batch_y, columns_to_observe)

                pred = batch_y.detach().cpu().numpy()
                true = outputs.detach().cpu().numpy()
                outputs_o = outputs_o.detach().cpu().numpy()
                outputs_d = outputs_d.detach().cpu().numpy()
                batch_y_o = batch_y_o.detach().cpu().numpy()
                batch_y_d = batch_y_d.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                preds_o.append(outputs_o[:, :, 0, :])
                trues_o.append(batch_y_o[:, :, 0, :])
                preds_d.append(outputs_d[:, :, 0, :])
                trues_d.append(batch_y_d[:, :, 0, :])

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    random_i = random.randrange(0, self.args.num_pairs - 1)
                    gt = np.concatenate((input[0, :, random_i], true[0, :, random_i]), axis=0)
                    pd = np.concatenate((input[0, :, random_i], pred[0, :, random_i]), axis=0)
                    gt = scaler.inverse_transform(gt)
                    pd = scaler.inverse_transform(pd)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        trues = np.round(trues, decimals=4)
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, r_2 = metric(preds, trues)
        print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, mape:{:.4f}, r_2:{:.4f}'.format(mae, mse, rmse, mape, r_2))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, r_2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        preds_o = np.concatenate(preds_o, axis=0)
        trues_o = np.concatenate(trues_o, axis=0)
        preds_d = np.concatenate(preds_d, axis=0)
        trues_d = np.concatenate(trues_d, axis=0)
        preds_o = preds_o.reshape(-1, preds_o.shape[-2], preds_o.shape[-1])
        trues_o = trues_o.reshape(-1, trues_o.shape[-2], trues_o.shape[-1])
        preds_d = preds_d.reshape(-1, preds_d.shape[-2], preds_d.shape[-1])
        trues_d = trues_d.reshape(-1, trues_d.shape[-2], trues_d.shape[-1])

        mae_o, mse_o, rmse_o, mape_o, r_2_o = metric(preds_o, trues_o)
        print('mae_o:{:.4f}, mse_o:{:.4f}, rmse_o:{:.4f}, mape_o:{:.4f}, r_2_o:{:.4f}'.format(mae_o, mse_o, rmse_o, mape_o, r_2_o))
        np.save(folder_path + 'metrics_o.npy', np.array([mae_o, mse_o, rmse_o, mape_o, r_2_o]))
        np.save(folder_path + 'pred_o.npy', preds_o)
        np.save(folder_path + 'true_o.npy', trues_o)

        mae_d, mse_d, rmse_d, mape_d, r_2_d = metric(preds_d, trues_d)
        print('mae_d:{:.4f}, mse_d:{:.4f}, rmse_d:{:.4f}, mape_d:{:.4f}, r_2_d:{:.4f}'.format(mae_d, mse_d, rmse_d, mape_d, r_2_d))
        np.save(folder_path + 'metrics_d.npy', np.array([mae_d, mse_d, rmse_d, mape_d, r_2_d]))
        np.save(folder_path + 'pred_d.npy', preds_d)
        np.save(folder_path + 'true_d.npy', trues_d)

        f = open("log.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        for i in range(self.args.pred_len):
            preds_scale = preds[:, i, :]
            trues_scale = trues[:, i, :]
            mae, mse, rmse, mape, r_2 = metric(preds_scale, trues_scale)
            if i % 1 == 0:
                log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
                print(log.format(i + 1, mae, mape, rmse))
                f.write(log.format(i + 1, mae, mape, rmse) + "  \n")
        f.write('\n')
        f.close()
        return