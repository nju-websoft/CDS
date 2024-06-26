package method;

import HBLL.*;

import java.util.ArrayList;
import java.util.HashMap;

public class AnsTree {
    public int root;
    public HashMap<Integer, Integer> preNode;
    public HashMap<Integer, Integer> depthMap;
    public int kwNum;
    public ArrayList<Integer> kwNode;
    public double weight=0;;
    public static final double INF=0x3f3f3f3f;
    public ArrayList<Double> min_weight;
    public ArrayList<Long> time;

    public AnsTree(int root, int kwdNum, ArrayList<Integer> kwdNode)
    {
        this.root=root;
        preNode=new HashMap<Integer, Integer>(0);
        depthMap=new HashMap<Integer, Integer>(0);
        kwNum=kwdNum;
        kwNode=kwdNode;
    }

    public int depth(int terminal)
    {
        if(depthMap.containsKey(terminal))
        {
            return depthMap.get(terminal);
        }
        int depth=0;
        int pre=terminal;
        while(pre!=root)
        {
            if(preNode.containsKey(pre))
            {
                pre=preNode.get(pre);
                depth=depth(pre)+1;
                depthMap.put(terminal,depth);
                return depth;
            }
            else
            {
                return -1;
            }
        }
        return depth;
    }

    public void addTotalPath(TreeHubSub thb, HopLimitHL hl, TerminalItem item, WeightedGraph ww)
    {
        int depth=depth(item.freshOrigin);
        weight+=item.dis;
        //addPath(thb,hl,item.freshOrigin,item.mid,item.hop1-depth,ww);
        //addreversePath(thb,hl,item.terminal,item.mid,item.hop2,ww);
        addPath(thb,hl,item.freshOrigin,item.mid,item.hop1-depth,item.originHubLoc,ww);
        addreversePath(thb,hl,item.terminal,item.mid,item.hop2,item.terminalHubLoc,ww);
    }




    void addPath(TreeHubSub thb, HopLimitHL hl, int u, int v, int hop, int hopLoc, WeightedGraph ww)
    {
        if(hop==0)
        {
            return;
        }
        int pre=hl.preOfLabel[hopLoc];
        if(preNode.containsKey(pre))
        {
            int prepre=preNode.get(pre);
            for(Edge e:ww.graph.get(pre))
            {
                if(e.v==prepre)
                {
                    weight-=e.weight;
                }
            }
            preNode.replace(pre,prepre,u);
        }
        else
        {
            preNode.put(pre,u);
            thb.addterminal(hl,pre);
        }
        int i=hl.preOfLabelLoc[hopLoc];
        addPath(thb,hl,pre,v,hop-1,hl.preOfLabelLoc[hopLoc],ww);
    }


    //u is terminal,v is mid
    void addreversePath(TreeHubSub thb, HopLimitHL hl, int u, int v, int hop, int hopLoc, WeightedGraph ww)
    {
        if(hop==0)
        {
            return;
        }
        int pre=hl.preOfLabel[hopLoc];
        addreversePath(thb,hl,pre,v,hop-1,hl.preOfLabelLoc[hopLoc],ww);
        if(preNode.containsKey(u))
        {
            int prepre=preNode.get(u);
            for(Edge e:ww.graph.get(u))
            {
                if(e.v==prepre)
                {
                    weight-=e.weight;
                }
            }
            preNode.replace(u,prepre,pre);
        }
        else
        {
            preNode.put(u,pre);
            thb.addterminal(hl,u);
        }
        return;
    }


    public int diameter()
    {
        int d=0;
        int[] depth=new int[kwNode.size()];
        for(int i=0;i<kwNode.size();i++)
        {
            int cur=kwNode.get(i);
            while(cur!=root)
            {
                cur=preNode.get(cur);
                depth[i]++;
            }
            if(d<depth[i])
            {
                d=depth[i];
            }
        }

        for(int i=0;i<kwNode.size()-1;i++)
        {
            for(int j=i+1;j<kwNode.size();j++)
            {
                int d1=depth[i],d2=depth[j];
                int cur1=kwNode.get(i),cur2=kwNode.get(j);
                while(d1>d2)
                {
                    cur1=preNode.get(cur1);
                    d1--;
                }
                while(d2>d1)
                {
                    cur2=preNode.get(cur2);
                    d2--;
                }
                while(cur1!=cur2)
                {
                    cur1=preNode.get(cur1);
                    cur2=preNode.get(cur2);
                    d1--;
                    d2--;
                }
                int dtemp=depth[i]+depth[j]-d1-d2;
                if(d<dtemp)
                {
                    d=dtemp;
                }
            }
        }

        return d;
    }





}
