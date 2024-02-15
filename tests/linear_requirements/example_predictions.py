import torch
import pickle as pkl


def example_predictions_url():
    # predictions = torch.tensor([-10.0, 5.0, -2.0, -9, 2, 20, -1]).unsqueeze(0)  # constraints_disj1: Corrected predictions tensor([-10.,  10.,  -2.,   3.,   1.,   2.,  -1.], grad_fn=<CopySlices>) True
    # y_38 y_37 y_21 y_3 y_31 y_26 y_28 y_2 y_19 y_25 y_23 y_13 y_20 y_1 y_4 y_5 y_6 y_7 y_8 y_9 y_10 y_11 y_12 y_14 y_15 y_16 y_17 y_0
    y_18 = y_22 = y_24 = y_27 = y_29 = y_30 = y_32 = y_33 = y_34 = y_35 = y_36 = -100
    y_38 = 0
    y_37 = 0
    y_21 = -1
    y_3 = 1
    y_31 = -3
    y_26 = 5
    y_28 = -3
    y_2 = 2
    y_19 = -4
    y_25 = -111 # change to -1 , value should be corrected to >0
    y_23 = 0
    y_13 = 0
    y_20 = 0
    y_1 = 0
    y_4 = 0
    y_5 = 0
    y_6 = 0
    y_7 = 0
    y_8 = 0
    y_9 = 0
    y_10 = 0
    y_11 = 0
    y_12 = 0
    y_14 = 0
    y_15 = 0
    y_16 = 0
    y_17 = 0
    y_0 = -14.0
    p1 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(39)])+']')).unsqueeze(0)   # constraints_disj1: Corrected predictions tensor([-10.,  10.,  -2.,   3.,   1.,   2.,  -1.], grad_fn=<CopySlices>) True

    y_2 = -18
    p2 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(39)])+']')).unsqueeze(0)
    predictions = torch.cat([p1,p1,p2,p1,p2],dim=0)
    return predictions


def example_predictions_botnet():
    # preds = pkl.load(open('../data/botnet/example_botnet_preds', 'rb'))
    preds = pkl.load(open('../../data/linear_requirements/botnet/example_predictions.pkl', 'rb'))
    original_preds = preds['original'].detach()
    corrected_preds = preds['corrected'].detach()
    original_preds = original_preds.clamp(-1e3, 1e3)
    print('Loaded preds with shape', original_preds.shape)
    return original_preds
