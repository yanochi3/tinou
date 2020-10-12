import java.io.*;


public class Main {
	public static void main(String arg[]) {
		try {    // �t�@�C���ǂݍ��݂Ɏ��s�������̗�O�����̂��߂�try-catch�\��
            String fileName = "dataset_example.txt"; // �t�@�C�����w��

            // �����R�[�hUTF-8���w�肵��BufferedReader�I�u�W�F�N�g�����
            BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
            String args = "";
            for(int i = 0; i< arg.length; i++) 
            {
            	args += arg[i]+" ";
            }
            // �ϐ�line��1�s���ǂݍ���for��
            for (String line = in.readLine(); line != null; line = in.readLine()) {
		            	if((new Unifier()).unify(args,line) == true) 	
		            	{
		            	System.out.println(line);
		            	}
            }
            in.close();  // �t�@�C�������
        } catch (IOException e) {
            e.printStackTrace(); // ��O�������������܂ł̃X�^�b�N�g���[�X��\��
        }
	}
}
