from models.nn_model_v2 import check_all_attr, dataload_preprocessing
from models.nn_model_v2 import Net, Edu_Data
from models.nn_model_v2 import criterion
from models.nn_model_v2 import train, validate, test, best_validation_loss
from models.nn_model_v2 import show_data,plot_decision_regions_2class
import time, torch
from torch.utils.data import DataLoader, random_split

""" 변경해볼만한 변수들
1. non_use_attr: 주어진 데이터셋에서 학습에 활용하지 않을 설명변수명을 입력 -> 설명변수 조건 다양화, 성능과 큰 관련을 갖는 설명변수를 찾을 수 있음.
2. epochs: 모델이 전체 데이터셋을 학습하는 횟수 -> 큰 값으로 설정할수록 훈련시간 증가
3. batch_size: 모델이 학습할때 한번에 모든 데이터를 학습하는게 아니라, 분할해 학습하게 됩니다. 이때 분할의 유닛사이즈를 의미합니다. 
                -> 값이 작을수록 세밀하게 데이터를 학습할 수 있지만, 세밀한 학습이 정확도 향상을 항상 보장하진 않습니다.
4. optimizer: 모델의 추론 오류를 개선(반영)하는 방식 -> 다른 훈련파라미터들과 함께 조합해 사용. SGD, ADAM, AdaDelta 가 일반적으로 좋은 성능을 냄
"""

# data loading and preprocessing
# check_all_attr() # 주석을 해제해 모든 데이터 특성이름을 확인하세요.
non_use_attr =['age','agesq']
# non_use_attr = [
#     'g11ses' , 'g9ses' , 'g11math',
#     'g9math' , 'hs_par' , 'lfaminc0811' , 'twoparguar' , 'singlepar',
#     'lesshalf' , 'g9nonnative' , 'g9partalkclg' , 'g9schtalkclg' , 'g11catholic',
#     'g11otherprivate' , 'g9catholic' , 'g9otherprivate' , 'g11city' , 'g11suburb',
#     'g11town' , 'g11northeast' , 'g11midwest' , 'g11south' , 'g9city',
#     'g9suburb' , 'g9town' , 'g9northeast' , 'g9midwest' , 'g9south',
#     'm_s2controlsborn_yes' , 's2mcontrolsborn_yes' , 'm_s2controlsstoptrying_yes' , 's2mcontrolsstoptrying_yes' , 'm_s2controlschallenge_yes' ,      
#     's2mcontrolschallenge_yes' , 'repeatg9' , 'high_sci_nosci' , 'high_sci_gensci' , 'high_sci_spesci',
#     'high_sci_advsci' , 'high_sci_apib' , 'exp9ed_dontknow' , 'exp9ed_hs' , 'exp9ed_aa',
#     'exp9ed_ba' , 'exp9ed_grad' , 'exp11ed_hsbroad' , 'exp11ed_aa' , 'exp11ed_ba',
#     'exp11ed_grad' , 'req_ed_occ09_notknow' , 'reqed_occ09_hs' , 'reqed_occ09_aa' , 'reqed_occ09_ba',
#     'reqed_occ09_grad' , 'req_ed_occ12_notknow' , 'reqed_occ12_hs' , 'reqed_occ12_aa' , 'reqed_occ12_ba',
#     'reqed_occ12_grad']
X, y = dataload_preprocessing(non_use_attr)
#==

# preparation for training/testing
batch_size = 32

data_set = Edu_Data(X,y)
train_size = int(len(X) * 0.7)
val_size = int(train_size * 0.15)
test_size = len(X) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(data_set, lengths=[train_size, val_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
#==

# model params declaration
input_dim = len(X.columns)
hidden_dim = 512
dropout_r = 0.1
output_dim = 2
model = Net(input_dim, hidden_dim, output_dim, dropout=dropout_r)
#==

# Training process
epochs = 30 # The number of repeation for training whole dataset
learning_rate = 0.0001 # 
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)


for epoch in range (epochs):
    time_start = time.time()

    train_loss, train_accuracy = train(model, criterion, train_loader, optimizer)
    validation_loss, validation_accuracy = validate(model, val_loader)

    time_end = time.time()
    time_delta = time_end - time_start

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(model, 'nnmodel.pt')

    print(f'epoch number: {epoch + 1} | time elapsed: {time_delta}s')
    print(f'training loss: {train_loss:.3f} |  training accuracy: {train_accuracy * 100:.2f}%')
    print(f'validation loss: {validation_loss:.3f} |  validation accuracy: {validation_accuracy * 100:.2f}%')
    print()
#==

# best result
best_model = torch.load('nnmodel.pt')
test_loss, test_accuracy, result = test(best_model, test_loader)
print(f'Test loss: {test_loss:.3f} | test: {test_accuracy * 100:.2f}%')
#==

# data analyses
show_data(X, y)
plot_decision_regions_2class(X, y)
#==
