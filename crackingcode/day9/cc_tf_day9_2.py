
# coding: utf-8

# In[ ]:


# How to Retrain Inception's Final Layer for New Categories

# https://www.flickr.com/
#cd ~
#curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
#tar xzf flower_photos.tgz

#bazel build tensorflow/examples/image_retraining:retrain


# fixing bazel version issue, no larger than bazel_0.5.4
# https://github.com/bazelbuild/bazel/releases
# https://docs.bazel.build/versions/master/install-os-x.html
#chmod +x bazel-0.5.4-installer-darwin-x86_64.sh 
#./bazel-0.5.4-installer-darwin-x86_64.sh --user
#export PATH="$PATH:$HOME/bin"


# go to root fo tensorflow source code
#cd /Users/lipingzhang/Desktop/program/jd/tensorflow/tensorflow
#bazel build tensorflow/examples/image_retraining:retrain


#bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/flower_photos

#tensorboard --logdir /tmp/retrain_logs
#http://0.0.0.0:6006/


bazel build tensorflow/examples/label_image:label_image && bazel-bin/tensorflow/examples/label_image/label_image --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --output_layer=final_result --image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg
--input_layer=Mul

# --use_saved_model=false


# https://www.jefftk.com/p/detecting-tanks

