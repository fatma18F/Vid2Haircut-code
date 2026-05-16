import os
from shutil import copyfile

def file_backup(savepath, conf_path, dir_lis):
        os.makedirs(f"{savepath}", exist_ok=True)
               
        for dir_name in os.listdir(dir_lis):
            
            
                
            cur_dir = os.path.join(savepath, dir_name)
            
            if os.path.isdir(os.path.join(dir_lis, dir_name)):
                

                files = os.listdir(os.path.join(dir_lis, dir_name))
                for f_name in files:
                    if f_name[-3:] == '.py':
                        os.makedirs(cur_dir, exist_ok=True)
                        copyfile(os.path.join(dir_lis, dir_name, f_name), os.path.join(cur_dir, f_name))
           
            else:
                if cur_dir[-3:] == '.py':
                    copyfile(os.path.join(dir_lis, dir_name), cur_dir)
                                   
        copyfile(conf_path, os.path.join(savepath, 'config.conf'))
        
        