with self.graph.as_default():####默认图与自定义图的关系
    ckpt = tf.train.get_checkpoint_state(self.savefile)
       if ckpt and ckpt.model_checkpoint_path:
           print(''.join([ckpt.model_checkpoint_path,'.meta']))
           self.saver = tf.train.import_meta_graph(''.join([ckpt.model_checkpoint_path,'.meta']))
           self.saver.restore(self.session,ckpt.model_checkpoint_path)
       #print all variable
       for op in self.graph.get_operations():
       print(op.name, " " ,op.type)
       #返回模型中的tensor
       layers = [op.name for op in self.graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
       layers = [op.name for op in self.graph.get_operations()]
       feature_nums = [int(self.graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
       for feature in feature_nums:
            print(feature)

     '''restore tensor from model'''
     w_out = self.graph.get_tensor_by_name('W:0')
     b_out = self.graph.get_tensor_by_name('b:0')
     _input = self.graph.get_tensor_by_name('x:0')
     _out = self.graph.get_tensor_by_name('y:0')
     y_pre_cls = self.graph.get_tensor_by_name('output:0')
     #self.session.run(tf.global_variables_initializer())   ####非常重要，不能添加这一句
        pred = self.session.run(y_pre_cls,feed_dict={_input:_X})
        return pred
#saver = tf.train.import_meta_graph('my-model-1000.meta')

# tf.train.export_meta_graph
# tf.train.import_meta_graph