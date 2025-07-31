import torch

from torch.utils.data import DataLoader

from ttstl.backbone import Backbone
from ttstl.architectures.cnn_lstm import CNNLSTM
from ttstl.datasets.cnn_lstm_dataset import CNNLSTMDataset
from ttstl.utils.data_downloader import import_to_pandas
from ttstl.utils.data_preprocessor import get_dataset_cnn_lstm


def main():

    # Initialize the CNN-LSTM model
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTM().to(device)

    # Specify your credentials
    credentials_path = 'credentials.json'

    # Specify your bucket & blob
    bucket_name = 'evolving-microstructural-tk-data'
    blob_name_train = "data.nosync/Recorded_Synthetic_Data/01102025_short_event_driven_recorded_lob_1.parquet"
    blob_name_validation = "data.nosync/Recorded_Synthetic_Data/01102025_short_event_driven_recorded_lob_2.parquet"

    # Download from the bucket using credentials
    df_train = import_to_pandas(
        blob_name=blob_name_train,
        bucket_name=bucket_name,
        credentials_path=credentials_path)
    
    df_validation = import_to_pandas(
        blob_name=blob_name_validation,
        bucket_name=bucket_name,
        credentials_path=credentials_path)

    dataset_train = get_dataset_cnn_lstm(df_train, delta=100, quantile=None)
    dataset_validation = get_dataset_cnn_lstm(df_validation, delta=100, quantile=None)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=4)
    validation_loader = DataLoader(dataset_validation, batch_size=32, shuffle=False, num_workers=4)

    backbone = Backbone(model=model, delta_adapter=100000.0)

    print(backbone.model)
    print(backbone.device)

    criterion = torch.nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # backbone.train(train_loader, 
    #                criterion, 
    #                optimizer, 
    #                validation_loader=validation_loader,
    #                epochs=2)
    
    blob_name_ptlob = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_driven_recorded_lob_0.parquet'
    blob_name_ptme = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_market_event_data_0.parquet'

    df_ptlob = import_to_pandas(
        blob_name=blob_name_ptlob,
        bucket_name=bucket_name,
        credentials_path=credentials_path)
    
    df_ptme = import_to_pandas(
        blob_name=blob_name_ptme,
        bucket_name=bucket_name,
        credentials_path=credentials_path)
    
    print(df_ptlob.info())
    print(df_ptme.info())
    
    print(df_ptlob.head())
    print(df_ptme.head())

    
    df_pt = df_ptlob.merge(df_ptme, on = 'timestamp', how = 'inner')
    df_pt.set_index('timestamp', inplace=True)
    

    print(backbone._select_stationary_variables(df_pt, kpss_drop = True))
    

if __name__ == "__main__":
    main()
