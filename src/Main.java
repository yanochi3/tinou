import java.io.*;


public class Main {
	public static void main(String arg[]) {
		try {    // ファイル読み込みに失敗した時の例外処理のためのtry-catch構文
            String fileName = "dataset_example.txt"; // ファイル名指定

            // 文字コードUTF-8を指定してBufferedReaderオブジェクトを作る
            BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
            String args = "";
            for(int i = 0; i< arg.length; i++) 
            {
            	args += arg[i]+" ";
            }
            // 変数lineに1行ずつ読み込むfor文
            for (String line = in.readLine(); line != null; line = in.readLine()) {
		            	if((new Unifier()).unify(args,line) == true) 	
		            	{
		            	System.out.println(line);
		            	}
            }
            in.close();  // ファイルを閉じる
        } catch (IOException e) {
            e.printStackTrace(); // 例外が発生した所までのスタックトレースを表示
        }
	}
}
