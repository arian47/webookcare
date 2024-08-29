import pathlib

class PathStruct:
    def __init__(self, name=None):
        self.name = name
    
    def find_files(self, path, file_type):
        # dirs = [i for i in path.iterdir() if i.is_dir()]
        if file_type in ['pdf', 'docx',]:
            # foi = [j for i in dirs for j in i.rglob(f'*.{file_type}')]
            foi = [i for i in path.rglob(f'*.{file_type}')]
            # foi = [j for i in dirs for j in i.rglob(f'**/*') if j.contains('.docx')]
            return foi
        else:
            raise Exception('File type not handled!')
        
    def create_paths(self, file_type):
        base_dir = pathlib.Path(__file__).parent.absolute()
        if self.name=='service_recommendation':
            # prepare old_data
            self.parent_dir = base_dir.joinpath(
                "service_recommendation/data/old data"
            )
        elif self.name=='credentials_recommendation':
            self.parent_dir = base_dir.joinpath(
                "credentials_recommendation/data"
            )
        else:
            raise Exception
        # parent_dir = pathlib.Path(parent_dir)
        return self.find_files(self.parent_dir, file_type)