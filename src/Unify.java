/***
  Unify Program written



  �ϐ�:�O�ɁH������D  

  Examle:
  % Unify "Takayuki" "Takayuki"
  true

  % Unify "Takayuki" "Takoyuki"
  false

  % Unify "?x am Takayuki" "I am Takayuki"
  ?x = I .

  % Unify "?x is ?x" "a is b"
  false

  % Unify "?x is ?x" "a is a"
  ?x = a .

  % Unify "?x is a" "b is ?y"
  ?x = b.
  ?y = a.

  % Unify "?x is a" "?y is ?x"
  ?x = a.
  ?y = a.

  Unify �́C���j�t�B�P�[�V�����ƍ��A���S���Y�����������C
  �p�^�[���\�����r���Ė����̂Ȃ�����ɂ���ē���Ɣ��f
  �ł��邩�ǂ����𒲂ׂ�D

  �|�C���g�I
  �����ł́C�X�g�����O���m�̒P�ꉻ�ł��邩��C�o���������s���K�v�͂Ȃ��D
  �������C"?x is a"�Ƃ����\�L��"is(?x,a)"�Ƃ���ȂǁC�\�����g���Ȃ�΁C
  �P�ꉻ�ɂ����ďo���������s���K�v������D
  �Ⴆ�΁C"a(?x)"��"?x"��P�ꉻ����� ?x = a(a(a(...))) �ƂȂ�C
  �������[�v�Ɋׂ��Ă��܂��D

  ***/

import java.util.*;

class Unify {
    public static void main(String arg[]){
    if(arg.length != 2){
        System.out.println("Usgae : % Unify [string1] [string2]");
    } else {
        System.out.println((new Unifier()).unify(arg[0],arg[1]));
    }
    }
}

class Unifier {
    StringTokenizer st1;
    String buffer1[];    
    StringTokenizer st2;
    String buffer2[];
    HashMap<String,String> vars;
    
    Unifier(){
        vars = new HashMap<String,String>();
    }

    public boolean unify(String string1,String string2,HashMap<String,String> bindings){
        this.vars = bindings;
        if(unify(string1,string2)){
    	    return true;
    	} else {
    	    return false;
    	}
    }

    public boolean unify(String string1,String string2){
        //System.out.println(string1);
        //System.out.println(string2);
    
        // �����Ȃ琬��
        if(string1.equals(string2)) return true;
    
        // �e�X�g�[�N���ɕ�����
        st1 = new StringTokenizer(string1);
        st2 = new StringTokenizer(string2);
    
        // �����قȂ����玸�s
        if(st1.countTokens() != st2.countTokens()) return false;
    
        // �萔���m
        int length = st1.countTokens();
        buffer1 = new String[length];
        buffer2 = new String[length];
        for(int i = 0 ; i < length; i++){
            buffer1[i] = st1.nextToken();
            buffer2[i] = st2.nextToken();
        }
        for(int i = 0 ; i < length ; i++){
            if(!tokenMatching(buffer1[i],buffer2[i])){
                return false;
            }
        }
    
    
        // �Ō�܂� O.K. �Ȃ琬��
        //System.out.println(vars.toString());	//�o�͂͂���Ȃ��̂ŃR�����g�A�E�g
        return true;
    }

    boolean tokenMatching(String token1,String token2){
        if(token1.equals(token2)) return true;
        if( var(token1) && !var(token2)) return varMatching(token1,token2);
        if(!var(token1) &&  var(token2)) return varMatching(token2,token1);
        if( var(token1) &&  var(token2)) return varMatching(token1,token2);
        return false;
    }

    boolean varMatching(String vartoken,String token){
        if(vars.containsKey(vartoken)){
            if(token.equals(vars.get(vartoken))){
                return true;
            } else {
                return false;
            }
        } else {
            replaceBuffer(vartoken,token);
            if(vars.containsValue(vartoken)){
                replaceBindings(vartoken,token);
            }
            vars.put(vartoken,token);
        }
        return true;
    }

    void replaceBuffer(String preString,String postString){
        for(int i = 0 ; i < buffer1.length ; i++){
            if(preString.equals(buffer1[i])){
                buffer1[i] = postString;
            }
            if(preString.equals(buffer2[i])){
                buffer2[i] = postString;
            }
        }
    }
    
    void replaceBindings(String preString,String postString){
    Iterator<String> keys;
    for(keys = vars.keySet().iterator(); keys.hasNext();){
        String key = (String)keys.next();
        if(preString.equals(vars.get(key))){
        vars.put(key,postString);
        }
    }
    }
    
    
    boolean var(String str1){
    // �擪�� ? �Ȃ�ϐ�
    return str1.startsWith("?");
    }
    
    String getVartoken(String key) {
    	return vars.get(key);
    }

}