exec(open("../mlp/mlp_augmentedb_categoricalcrossentropy_adam_normalized.py").read())                           #k-fold
exec(open("../mlp/mlp_fulldb_categoricalcrossentropy_adam.py").read())                                          #k-fold
exec(open("../mlp/mlp_reducedb_categoricalcrossentropy_adam_normalized.py").read())                             #k-fold
exec(open("../mlp/mlp_fulldb_meansquarederror_adam.py").read())                                                 #k-fold
exec(open("../mlp/mlp_reducedb_categoricalcrossentropy_adam.py").read())                                        #k-fold
exec(open("../mlp/mlp_reducedb_meansquarederror_sgd.py").read())                                                #k-fold
exec(open("../mlp/mlp_fulldb_categoricalcrossentropy_adam_normalized.py").read())                               #k-fold
exec(open("../mlp/mlp_fulldb_meansquarederror_adam_normalized.py").read())                                      #k-fold
exec(open("../mlp/mlp_fulldb_categoricalhinge_adam_normalized.py").read())                                      #k-fold
exec(open("../mlp/mlp_fulldb_categoricalhinge_adam.py").read())                                                 #k-fold
exec(open("../mlp/mlp_fulldb_hinge_adam_normalized.py").read())                                                 #k-fold
exec(open("../mlp/mlp_nohumiditydb_categoricalcrossentropy_adam.py").read())                                    #k-fold
exec(open("../mlp/mlp_nohumiditydb_meansquarederror_adam_normalized.py").read())                                #k-fold
exec(open("../mlp/mlp_nohumiditydb_categoricalcrossentropy_adam_normalized.py").read())                         #k-fold
exec(open("../mlp/mlp_nohumiditydb_meansquarederror_adam.py").read())                                           #k-fold
exec(open("../mlp/mlp_secondaugmentedb_categoricalcrossentropy_adam.py").read())                                #k-fold
exec(open("../mlp/mlp_secondaugmentedb_meansquarederror_adam.py").read())                                       #k-fold

exec(open("../lstm/lstm_fulldb_categoricalcrossentropy_adam.py").read())                                        #k-fold
exec(open("../lstm/lstm_reducedb_categoricalcrossentropy_adam.py").read())                                      #k-fold
exec(open("../lstm/lstm_fulldb_meansquarederror_adam.py").read())                                               #k-fold
exec(open("../lstm/lstm_nohumiditydb_categoricalcrossentropy_adam.py").read())                                  #k-fold
exec(open("../lstm/lstm_nohumiditydb_categoricalcrossentropy_adam_bidirectional.py").read())                    #k-fold
exec(open("../lstm/lstm_nohumiditydb_categoricalcrossentropy_adam_normalized_bidirectional.py").read())         #k-fold
exec(open("../lstm/lstm_reducedb_categoricalcrossentropy_adam_bidirectional.py").read())                        #k-fold
exec(open("../lstm/lstm_secondaugmentedb_categoricalcrossentropy_adam.py").read())                              #k-fold
exec(open("../lstm/lstm_secondaugmentedb_categoricalcrossentropy_adam_bidirectional.py").read())                #k-fold
#
# exec(open("../cnn/cnn_fulldb_categoricalcrossentropy_adam_refined.py").read())
# exec(open("../cnn/cnn_fulldb_categoricalcrossentropy_adam_refined_normalized.py").read())
# exec(open("../cnn/cnn_fulldb_categoricalcrossentropy_adam.py").read())
# exec(open("../cnn/cnn_fulldb_meansquarederror_adam_refined_normalized.py").read())
# exec(open("../cnn/cnn_fulldb_categoricalcrossentropy_adam_refined_greater_normalization.py").read())
# exec(open("../cnn/cnn_fulldb_categoricalcrossentropy_adam_refined_greatest_normalization.py").read())
# exec(open("../cnn/cnn_fulldb_categoricalcrossentropy_adam_refined_smaller_normalization.py").read())
# exec(open("../cnn/cnn_augmentedb_categoricalcrossentropy_adam.py").read())
# exec(open("../cnn/cnn_nohumiditydb_meansquarederror_adam_refined_normalized.py").read())
exec(open("../cnn/cnn_secondaugmentedb_categoricalcrossentropy_adam_refined.py").read())
exec(open("../cnn/cnn_secondaugmentedb_categoricalcrossentropy_adam.py").read())
#
exec(open("../resnet/resnet_fulldb_categoricalcrossentropy_adam.py").read())                                  #k-fold
exec(open("../resnet/resnet_fulldb_meansquarederror_adam_refined_normalized.py").read())                      #k-fold
exec(open("../resnet/resnet_secondaugmentedb_categoricalcrossentropy_adam.py").read())                        #k-fold
