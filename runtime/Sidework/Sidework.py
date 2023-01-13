import pandas as pd
# swift_train = pd.read_csv('dev_swift_transaction_train_dataset.csv')
# swift_train['Label'] = swift_train['Label'].astype(int)
# swift_train.to_csv('dev_swift_transaction_train_dataset.csv')
# swift_test = pd.read_csv('dev_swift_transaction_test_dataset.csv')
# swift_test['Label'] = swift_test['Label'].astype(int)
# swift_test.to_csv('dev_swift_transaction_test_dataset.csv')
bank = pd.read_csv('dev_bank_dataset.csv')
bank['Flags'] = bank['Flags'].astype(int)
