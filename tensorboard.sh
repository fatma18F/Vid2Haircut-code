#on remote
tensorboard --logdir=/home/ayed/monocular-hair-modeling/outputs/exps_inverse_stage_vanessa0/000001/logs --port=6006

#on local
ssh -L 6006:localhost:6006  ayed@SWS04.vc.cit.tum.de
